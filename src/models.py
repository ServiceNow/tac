import torch

from torch.cuda.amp import autocast, GradScaler
from torch.optim import *
from torch.optim.lr_scheduler import *

from src.utils import (
    compute_loss,
    predict,
    SimCoef,
    perturb_data,
    set_device,
)
from src.TAC import (
    LeNet_TAC,
    Wide_ResNet_TAC,
)
from src.constants import DEVICE
from src.codes import get_code, UniqueBinaryVectorsMaker
import foolbox as fb
from sklearn import metrics
import tqdm


def get_trainable_parameters(model):
    # Always trains projection parameters.
    trainable_parameters = list(model.projection_params.parameters())
    if model.train_classification_head:
        trainable_parameters.extend(list(model.classification_head.parameters()))
    if model.train_base_model:
        trainable_parameters.extend(list(model.base_model.parameters()))
    return trainable_parameters


def get_projection_params(input_dimensions, mode, n_slices):

    assert mode in {"none", "small", "large", "very_large", "x_large", "2x_large"}

    layer_list = []

    for in_dimension in input_dimensions:
        if mode == "none":
            layer = torch.nn.Identity()
        elif mode == "small":
            layer = torch.nn.Linear(in_dimension, in_dimension)
        elif mode == "large":
            layer = torch.nn.Sequential(
                torch.nn.Linear(in_dimension, in_dimension),
                torch.nn.ReLU(),
                torch.nn.Linear(in_dimension, in_dimension),
            )
        elif mode == "very_large":
            layer = torch.nn.Sequential(
                torch.nn.Linear(in_dimension, in_dimension),
                torch.nn.ReLU(),
                torch.nn.Linear(in_dimension, in_dimension),
                torch.nn.ReLU(),
                torch.nn.Linear(in_dimension, in_dimension),
            )
        elif mode == "x_large":
            layer = torch.nn.Sequential(
                torch.nn.LayerNorm(in_dimension),
                torch.nn.Linear(in_dimension, in_dimension),
                torch.nn.ReLU(),
                torch.nn.Linear(in_dimension, in_dimension),
                torch.nn.ReLU(),
                torch.nn.Linear(in_dimension, in_dimension),
                torch.nn.LayerNorm(in_dimension),
            )
        else:
            layer = torch.nn.Sequential(
                torch.nn.LayerNorm(in_dimension),
                torch.nn.Linear(in_dimension, in_dimension),
                torch.nn.ReLU(),
                torch.nn.Linear(in_dimension, in_dimension),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(in_dimension),
                torch.nn.Linear(in_dimension, n_slices),
                torch.nn.ReLU(),
                torch.nn.Linear(n_slices, n_slices),
                torch.nn.ReLU(),
                torch.nn.Linear(n_slices, in_dimension),
                torch.nn.LayerNorm(in_dimension),
            )
        layer_list.append(layer)

    return torch.nn.ModuleList(layer_list)


def get_model(exp_dict):

    if "attacker_config" not in exp_dict:
        exp_dict["attacker_config"] = None

    # Try Hadamard codes first.
    codes_matrix = get_code(
        exp_dict["num_classes"],
        exp_dict["num_classifiers"],
    )

    num_slices = int(codes_matrix.size(1) / exp_dict["num_classifiers"])

    # If codes are too long, get random codes matching the maximum length.
    if num_slices > exp_dict["max_num_of_slices_per_layer"]:
        codes_matrix = get_code(
            exp_dict["num_classes"],
            exp_dict["num_classifiers"],
            UniqueBinaryVectorsMaker(exp_dict["max_num_of_slices_per_layer"]),
        )
        num_slices = exp_dict["max_num_of_slices_per_layer"]

    codes_matrix.requires_grad = False

    if "codes_noise_level" not in exp_dict:
        exp_dict["codes_noise_level"] = 0.0

    model = TAC(
        model_name=exp_dict["model_name"],
        num_slices=num_slices,
        use_mixup=exp_dict["use_mixup"],
        codes_noise_level=exp_dict["codes_noise_level"],
        slicing_strategy=exp_dict["slicing_strategy"],
        attacker_config=exp_dict["attacker_config"],
        max_grad_norm=exp_dict["grad_clip"],
        bin_loss_scale=exp_dict["bin_loss_scale"],
        ce_loss_scale=exp_dict["ce_loss_scale"],
        train_classification_head=exp_dict["train_classification_head"],
        train_base_model=exp_dict["train_base_model"],
        use_mixed_precision=exp_dict["use_mixed_precision"],
    )

    assert exp_dict["num_classifiers"] == len(model.base_model.feature_dims)

    model.codes_matrix = codes_matrix

    model.classification_head = torch.nn.Linear(
        model.base_model.feature_dims[-1], codes_matrix.size(0)
    )

    model.similarities_reg_coef = SimCoef(exp_dict["similarities_reg_coef"])

    model.projection_params = get_projection_params(
        model.base_model.feature_dims, exp_dict["projection_type"], num_slices
    )

    model.opt = globals()[exp_dict["optimizer_config"]["name"]](
        get_trainable_parameters(model),
        **exp_dict["optimizer_config"]["kwargs"],
    )

    model.scheduler = globals()[exp_dict["scheduler_config"]["name"]](
        model.opt, **exp_dict["scheduler_config"]["kwargs"]
    )

    return model


# =====================================================


class TAC(torch.nn.Module):
    def __init__(
        self,
        model_name,
        num_slices,
        use_mixup,
        codes_noise_level,
        slicing_strategy,
        attacker_config=None,
        max_grad_norm=100.0,
        bin_loss_scale=1.0,
        ce_loss_scale=1.0,
        train_classification_head=False,
        train_base_model=False,
        use_mixed_precision=False,
    ):
        super().__init__()

        self.base_model = globals()[model_name](
            num_slices=num_slices,
            slicing_strategy=slicing_strategy,
        )

        self.use_mixup = use_mixup
        self.codes_noise_level = codes_noise_level
        self.best_val_score = -float("inf")
        self.max_grad_norm = max_grad_norm
        self.bin_loss_scale = bin_loss_scale
        self.ce_loss_scale = ce_loss_scale
        self.train_classification_head = train_classification_head
        self.train_base_model = train_base_model
        self.use_mixed_precision = use_mixed_precision

        if attacker_config is not None:
            self.attacker = fb.attacks.LinfPGD(
                abs_stepsize=attacker_config["attack_budget"]
                / attacker_config["attack_steps"],
                steps=attacker_config["attack_steps"],
                random_start=True,
            )

            self.attacker_config = attacker_config
        else:
            self.attacker = None

        if self.use_mixed_precision:
            self.scaler = GradScaler()

    def train_on_batch(self, batch, epoch, **extras):

        data, target = batch
        data, target = set_device(DEVICE, data, target)
        self.codes_matrix = self.codes_matrix.to(DEVICE, non_blocking=True)

        if self.attacker is not None:
            data = perturb_data(
                data,
                target,
                self,
                self.attacker,
                self.attacker_config["attack_budget"],
                self.attacker_config["image_bounds"],
                baseline_mode=self.attacker_config["attack_classification_head"],
            )

        self.opt.zero_grad()
        self.base_model.eval()
        if self.train_classification_head:
            self.classification_head.train()
        else:
            self.classification_head.eval()
        self.projection_params.train()

        if self.use_mixed_precision:
            with autocast():
                loss_ce, loss_bin, features = compute_loss(
                    self,
                    data,
                    target,
                    self.codes_matrix,
                    sim_coef=self.similarities_reg_coef,
                    mixup=self.use_mixup,
                    codes_noise_level=self.codes_noise_level,
                    return_logits=False,
                    return_raw_features=True,
                )
                if self.train_classification_head:
                    features = features[-1].detach()
                    if features.dim() > 2:
                        features = features.view(
                            [features.size(0), features.size(1), -1]
                        ).mean(-1)

                    classification_head_output = self.classification_head(features)

                    classification_head_loss = torch.nn.CrossEntropyLoss()(
                        classification_head_output, target
                    )
        else:
            loss_ce, loss_bin, features = compute_loss(
                self,
                data,
                target,
                self.codes_matrix,
                sim_coef=self.similarities_reg_coef,
                mixup=self.use_mixup,
                codes_noise_level=self.codes_noise_level,
                return_logits=False,
                return_raw_features=True,
            )

            if self.train_classification_head:
                features = features[-1].detach()
                if features.dim() > 2:
                    features = features.view(
                        [features.size(0), features.size(1), -1]
                    ).mean(-1)

                classification_head_output = self.classification_head(features)

                classification_head_loss = torch.nn.CrossEntropyLoss()(
                    classification_head_output, target
                )

        training_loss = self.ce_loss_scale * loss_ce + self.bin_loss_scale * loss_bin
        if self.train_classification_head:
            training_loss += classification_head_loss

        if self.use_mixed_precision:
            self.scaler.scale(training_loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.opt.step()

        training_scores = {
            "train_loss": training_loss.item(),
            "train_loss_ce": loss_ce.item(),
            "train_loss_bin": loss_bin.item(),
        }

        if self.train_classification_head:
            training_scores[
                "classification_head_train_loss"
            ] = classification_head_loss.item()

        return training_scores

    def train_on_loader(self, loader, epoch, **extras):
        for batch in tqdm.tqdm(loader, desc="Epoch %d" % epoch, leave=False):
            train_dict = self.train_on_batch(batch, epoch)

        self.scheduler.step()

        train_dict.update({"lr": self.opt.param_groups[0]["lr"]})

        return train_dict

    def eval_on_loader(self, loader, **extras):
        self.base_model.eval()
        self.classification_head.eval()
        self.projection_params.eval()
        test_loss_ce = 0
        test_loss_bin = 0
        test_loss_classification_head = 0
        correct = 0
        correct_classification_head = 0
        inner_products_list = []
        mls_list = []
        error_detection_label_list = []
        n_batches = len(loader)
        for data, target in loader:
            data, target = set_device(DEVICE, data, target)
            self.codes_matrix = self.codes_matrix.to(DEVICE, non_blocking=True)
            if self.attacker is not None:
                data = perturb_data(
                    data,
                    target,
                    self,
                    self.attacker,
                    self.attacker_config["attack_budget"],
                    self.attacker_config["image_bounds"],
                    baseline_mode=self.attacker_config["attack_classification_head"],
                )
                self.projection_params.zero_grad()
            with torch.no_grad():
                (
                    current_test_loss_ce,
                    current_test_loss_bin,
                    output,
                    features,
                ) = compute_loss(
                    self,
                    data,
                    target,
                    self.codes_matrix,
                    sim_coef=self.similarities_reg_coef,
                    mixup=False,
                    codes_noise_level=0.0,
                    return_logits=True,
                    return_raw_features=True,
                )
                features = features[-1]
                if features.dim() > 2:
                    features = features.view(
                        [features.size(0), features.size(1), -1]
                    ).mean(-1)

                classification_head_output = self.classification_head(features)
                current_classification_head_loss = torch.nn.CrossEntropyLoss()(
                    classification_head_output, target
                )
                test_loss_ce += current_test_loss_ce.item()  # sum up batch loss
                test_loss_bin += current_test_loss_bin.item()  # sum up batch loss
                test_loss_classification_head += current_classification_head_loss.item()
                pred = predict(output, self.codes_matrix)[
                    0
                ]  # get the index with closest code
                correct += pred.eq(target.view_as(pred)).sum().item()
                pred_classification_head = classification_head_output.argmax(1)
                correct_classification_head += (
                    pred_classification_head.eq(
                        target.view_as(pred_classification_head)
                    )
                    .sum()
                    .item()
                )
                code_predictions = self.codes_matrix.index_select(
                    0, pred_classification_head
                )
                detection_label = (target == pred_classification_head).long().cpu()

            error_detection_label_list.append(detection_label)
            inner_products_list.append((output * code_predictions).sum(1).cpu())
            mls_list.append(classification_head_output.max(1)[0].cpu())

        inner_products = torch.cat(inner_products_list, 0)
        mls = torch.cat(mls_list, 0)
        error_detection_labels = torch.cat(error_detection_label_list, 0)

        fpr_cos, tpr_cos, _ = metrics.roc_curve(
            error_detection_labels, inner_products, pos_label=1
        )
        error_detection_auc = metrics.auc(fpr_cos, tpr_cos)
        error_detection_auc = max(error_detection_auc, 1.0 - error_detection_auc)

        fpr_mls, tpr_mls, _ = metrics.roc_curve(
            error_detection_labels, mls, pos_label=1
        )
        ed_auc_mls = metrics.auc(fpr_mls, tpr_mls)
        ed_auc_mls = max(ed_auc_mls, 1.0 - ed_auc_mls)

        return {
            "test_acc": correct / len(loader.dataset),
            "test_loss_ce": test_loss_ce / n_batches,
            "test_loss_bin": test_loss_bin / n_batches,
            "test_acc_classification_head": correct_classification_head
            / len(loader.dataset),
            "test_loss_classification_head": test_loss_classification_head / n_batches,
            "error_detection_auc": error_detection_auc,
            "ed_auc_mls": ed_auc_mls,
        }

    def eval_base_classifier_on_loader(self, loader, **extras):
        self.base_model.eval()
        self.classification_head.eval()
        test_loss = 0
        correct = 0
        n_batches = len(loader)
        for data, target in loader:
            data, target = set_device(DEVICE, data, target)
            with torch.no_grad():
                features = self.base_model.get_sets_of_features(data)[-1]
                if features.dim() > 2:
                    features = features.view(
                        [features.size(0), features.size(1), -1]
                    ).mean(-1)

                classification_head_output = self.classification_head(features)
                current_classification_head_loss = torch.nn.CrossEntropyLoss()(
                    classification_head_output, target
                )
                test_loss += current_classification_head_loss.item()
                pred_classification_head = classification_head_output.argmax(1)
                correct += (
                    pred_classification_head.eq(
                        target.view_as(pred_classification_head)
                    )
                    .sum()
                    .item()
                )

        return {
            "test_acc": correct / len(loader.dataset),
            "test_loss": test_loss / n_batches,
        }

    def forward(
        self,
        x,
        use_median=False,
        return_raw_features=False,
        tape_base_model_grads=False,
    ):
        return self.base_model(
            x,
            self.projection_params,
            use_median=use_median,
            tape_base_model_grads=self.train_base_model or tape_base_model_grads,
            return_raw_features=return_raw_features,
        )

    def forward_with_attention(
        self,
        x,
        tape_base_model_grads=False,
    ):
        return self.base_model.forward_with_attention(
            x,
            self.projection_params,
            tape_base_model_grads=self.train_base_model or tape_base_model_grads,
        )

    def get_state_dict(self):

        state_dict = {
            "projection_model": self.projection_params.state_dict(),
            "base_model": self.base_model.state_dict(),
            "classification_head": self.classification_head.state_dict(),
            "similarities_coef": self.similarities_reg_coef.state_dict(),
            "class_codes": self.codes_matrix.cpu(),
            "slicing_idx": self.base_model.slicing_idx,
            "opt": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_score": self.best_val_score,
        }

        return state_dict

    def set_state_dict(self, state_dict):
        self.projection_params.load_state_dict(state_dict["projection_model"])
        self.base_model.load_state_dict(state_dict["base_model"])
        self.classification_head.load_state_dict(state_dict["classification_head"])
        self.similarities_reg_coef.load_state_dict(state_dict["similarities_coef"])
        self.codes_matrix = state_dict["class_codes"]
        self.opt.load_state_dict(state_dict["opt"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.base_model.slicing_idx = state_dict["slicing_idx"]
        self.best_val_score = state_dict["best_val_score"]

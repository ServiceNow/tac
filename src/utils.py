import argparse
import torch
import numpy as np
import foolbox as fb
from src.losses import l1_loss, l1_ce_loss


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def make_mask(number_of_blocks, mask_length, device):

    dimension_per_block = mask_length // number_of_blocks

    slices_indices = (
        torch.arange(mask_length)
        .view(number_of_blocks, dimension_per_block)
        .long()
        .to(device)
    )

    masks = torch.zeros(
        (
            number_of_blocks,
            mask_length,
        ),
        device=device,
    )
    masks.scatter_(1, slices_indices, 1.0)

    return masks


def compute_detection_rate(fpr, tpr):
    fnr = 1 - tpr
    t = np.nanargmin(np.abs(fnr - fpr))
    return tpr[t]


def set_device(device, *args):
    output = []
    for el in args:
        if hasattr(el, "keys"):
            for k, v in el.items():
                if hasattr(v, "to"):
                    el[k] = v.to(device, non_blocking=True)
        else:
            el = el.to(device, non_blocking=True)
        output.append(el)

    return output


def interpolate_data_batch(
    data_batch, one_hot_labels, code_labels, interpolation_range=0.2
):
    """Mixup style data interpolation.

    Args:
        data_batch (torch.FloatTensor): batch of data with shape [batch_size, ...].
        one_hot_labels (torch.FloatTensor): batch of labels in one hot format.
        interpolation_range (float, optional): Concentration param for the Bet distribution.

    Returns:
        (torch.FloatTensor, torch.FloatTensor): batch of interpolated pairs from data_batch and codes.
    """

    try:
        data_batch = data_batch["pixel_values"]
        x_is_dict = True
    except IndexError:
        x_is_dict = False

    permutation_idx = torch.randperm(data_batch.size()[0], device=data_batch.device)

    data_pairs = data_batch[permutation_idx, ...]
    label_pairs = one_hot_labels[permutation_idx, ...]
    code_label_pairs = code_labels[permutation_idx, ...]

    # Mixup interpolator: random convex combination of pairs
    interpolation_factors = (
        torch.distributions.beta.Beta(interpolation_range, interpolation_range)
        .rsample(sample_shape=(data_pairs.size(0),))
        .to(data_batch.device)
    )
    # Create extra dimensions in the interpolation_factors tensor
    interpolation_factors_data = interpolation_factors[
        (...,) + (None,) * (data_batch.ndim - 1)
    ]
    interpolation_factors_labels = interpolation_factors[
        (...,) + (None,) * (one_hot_labels.ndim - 1)
    ]
    interpolation_factors_code_labels = interpolation_factors[
        (...,) + (None,) * (code_labels.ndim - 1)
    ]

    # Interpolation for a pair x_0, x_1 and factor t is given by t*(x_0)+(1-t)*x_1
    interpolated_batch = (
        interpolation_factors_data * data_batch
        + (1.0 - interpolation_factors_data) * data_pairs
    )
    interpolated_labels = (
        interpolation_factors_labels * one_hot_labels
        + (1.0 - interpolation_factors_labels) * label_pairs
    )
    interpolated_code_labels = (
        interpolation_factors_code_labels * code_labels
        + (1.0 - interpolation_factors_code_labels) * code_label_pairs
    )

    if x_is_dict:
        interpolated_batch = {"pixel_values": interpolated_batch}

    return interpolated_batch, interpolated_labels, interpolated_code_labels


def apply_noise_on_codes(codes, noise_level):
    noise_distribution = torch.distributions.Bernoulli(
        torch.ones_like(codes) * noise_level
    )
    noise_sample = -2.0 * noise_distribution.sample() + 1.0

    return codes * noise_sample


def compute_loss(
    model,
    data,
    labels,
    codes_matrix,
    sim_coef=None,
    mixup=False,
    codes_noise_level=0.0,
    return_logits=False,
    return_raw_features=False,
):

    labels_one_hot = torch.nn.functional.one_hot(
        labels, num_classes=codes_matrix.size(0)
    ).float()
    code_labels = codes_matrix.index_select(0, labels)

    if codes_noise_level > 0.0:
        codes_matrix = apply_noise_on_codes(codes_matrix, codes_noise_level)

    if mixup:
        with torch.no_grad():
            data, labels_one_hot, code_labels = interpolate_data_batch(
                data, labels_one_hot, code_labels
            )

    logits, features = model(data, return_raw_features=True)
    loss_bin = l1_loss(logits, code_labels)
    loss_ce = l1_ce_loss(
        logits, codes_matrix, labels_one_hot, sim_coef
    )

    to_return = [loss_ce, loss_bin]

    if return_logits:
        to_return.append(logits.detach())
    if return_raw_features:
        to_return.append(features)

    return to_return


def predict(logits, codes_matrix, mask=None):

    logits_signs = torch.sign(logits).float()

    if mask is not None:
        masked_logits = logits * mask.unsqueeze(0).repeat(logits.size(0), 1)
        masked_logits_signs = logits_signs * mask.unsqueeze(0).repeat(
            logits_signs.size(0), 1
        )
        masked_codes_matrix = codes_matrix * mask.unsqueeze(0).repeat(
            codes_matrix.size(0), 1
        )
    else:
        masked_logits = logits
        masked_logits_signs = logits_signs
        masked_codes_matrix = codes_matrix

    repeated_logits_signs = masked_logits_signs.unsqueeze(1).repeat(
        1, masked_codes_matrix.size(0), 1
    )

    repeated_codes_matrix = masked_codes_matrix.unsqueeze(0).repeat(
        masked_logits.size(0), 1, 1
    )

    pointwise_inner_products = (repeated_logits_signs * repeated_codes_matrix).sum(-1)

    predictions_full_codes = pointwise_inner_products.argmax(-1)

    confidence_scores = (masked_logits * codes_matrix[predictions_full_codes]).sum(-1)

    return predictions_full_codes, confidence_scores


def perturb_data(
    x,
    y,
    model,
    adversary,
    epsilon,
    bounds,
    baseline_mode=False,
):

    target_model = fb.PyTorchModel(
        wrapper(
            model.eval(),
            baseline_mode=baseline_mode,
        ).eval(),
        bounds=bounds,
    )

    batch_size = x.size(0)
    if batch_size > 1:
        # Only the first half of the batch will be perturbed
        clean_adv_idx = batch_size // 2
        _, x_adv, _ = adversary(
            target_model, x[:clean_adv_idx], y[:clean_adv_idx], epsilons=epsilon
        )
        x = torch.cat([x_adv, x[clean_adv_idx:]], 0)
    else:
        _, x, _ = adversary(target_model, x, y, epsilons=epsilon)

    return x


def retrieval_eval(embeddings, labels):

    # Compute cosine similarity matrix
    similarities = embeddings @ embeddings.T

    repeated_labels = labels.unsqueeze(1).repeat(1, similarities.size(0))

    top_indices = similarities.sort(1, descending=True).indices

    top_indices_labels = labels[top_indices]

    # Bool tensor indicating which rows contain the idx corresponding to the main diag. of the sim matrix
    hits = top_indices_labels.eq(repeated_labels)[:, 1:]

    r_at_1 = hits[:, :1].sum() / float(similarities.size(0))
    r_at_5 = hits[:, :5].sum(1).clamp(0.0, 1.0).sum() / float(similarities.size(0))

    ranks = hits.sort(1, descending=True).indices[:, 0] + 1.0
    mrr = (1 / ranks).mean()

    return r_at_1, r_at_5, mrr


class wrapper(torch.nn.Module):
    def __init__(
        self,
        base_model,
        baseline_mode=False,
        use_median=False,
    ):
        """Instantiates model wraping torch's model.

        This class defines a model wraping a base classifier in order to
        override its original forward() so that a single argument is passed.
        This is useful for computing adversaries

        Args:
            base_torch_model (torch.nn.Module): Torch model to be wrapped.
            baseline_mode (Bool): Whether the model is a baseline and should be used as such.
            use_median (Bool): Whether to use median reduction during inference
        """
        super(wrapper, self).__init__()

        self.base_model = base_model
        self.baseline_mode = baseline_mode
        self.use_median = use_median

    def forward(self, x):
        """Single argument forward. Used to compute adversaries.

        Args:
            x (torch.FloatTensor): Input data.

        Returns:
            torch.FloatTensor: Model outputs.
        """

        if not self.baseline_mode:
            self.base_model.codes_matrix = self.base_model.codes_matrix.to(x.device)
            logits = self.base_model(x, tape_base_model_grads=True)

            repeated_logits = logits.unsqueeze(1).repeat(
                1, self.base_model.codes_matrix.size(0), 1
            )

            repeated_codes_matrix = self.base_model.codes_matrix.unsqueeze(0).repeat(
                logits.size(0), 1, 1
            )

            # To create attacks, logits are the negative L1 distances between
            # the set of total activations and the codes.
            return -(repeated_logits - repeated_codes_matrix).abs().sum(-1)
        else:
            features = self.base_model.base_model.get_sets_of_features(x)[-1]
            features = features.view([features.size(0), features.size(1), -1]).mean(-1)
            outputs = self.base_model.classification_head(features)
            return outputs


class SimCoef(torch.nn.Module):
    def __init__(self, initial_value):
        super().__init__()
        self.coef = torch.nn.Parameter(torch.Tensor([initial_value]))

    def forward(self, sim_matrix):
        # Apply multiplicative factor on similarities
        # Clamping after exp to avoid numerical instabilities
        sim_matrix = sim_matrix * self.coef.clamp(1e-4, 50.0)

        return sim_matrix

ATTACKER_CONFIG_MNIST = {
    "image_bounds": [0.0, 1.0],
    "attack_budget": 0.3,
    "attack_steps": 10,
    "attack_classification_head": True,
}

ATTACKER_CONFIG_CIFAR10 = {
    "image_bounds": [0.0, 1.0],
    "attack_budget": 20.0 / 255.0,
    "attack_steps": 10,
    "attack_classification_head": True,
}

BASE_CONFIGS = {}
BASE_CONFIGS["mnist"] = {
    "dataset": "MNIST",
    "model_name": "LeNet_TAC",
    "slicing_strategy": "uniform",
    "num_classes": 10,
    "num_classifiers": 4,
    "max_num_of_slices_per_layer": 16,
    "train_batch_size": 8,
    "test_batch_size": 256,
    "optimizer_config": {
        "name": "SGD",
        "kwargs": {"lr": 1e-1, "momentum": 0.9, "weight_decay": 5e-5},
    },
    "scheduler_config": {
        "name": "MultiStepLR",
        "kwargs": {
            "milestones": [5, 100, 150],
            "gamma": 0.1,
        },
    },
    "grad_clip": 1.0,
    "use_mixup": False,
    "codes_noise_level": 0.15,
    "similarities_reg_coef": 10.0,  # Multiplicative factor applied on logits.
    "ce_loss_scale": 1.0,  # Scale factor applied on ce loss.
    "bin_loss_scale": 10.0,  # Scale factor applied on bin loss.
    "train_classification_head": True,
    "train_base_model": False,
    "n_workers": 2,
    "attacker_config": None,
    # "attacker_config": ATTACKER_CONFIG_MNIST,
    "projection_type": "small",
    "use_mixed_precision": False,
    "pretrained_path": "resources/mnist_model.p",
}

BASE_CONFIGS["cifar10"] = {
    "dataset": "CIFAR10",
    "model_name": "Wide_ResNet_TAC",
    "slicing_strategy": "uniform",
    "num_classes": 10,
    "num_classifiers": 3,
    "max_num_of_slices_per_layer": 16,
    "train_batch_size": 128,
    "test_batch_size": 128,
    "optimizer_config": {
        "name": "SGD",
        "kwargs": {"lr": 1e-2, "momentum": 0.9, "weight_decay": 5e-5},
    },
    "scheduler_config": {
        "name": "MultiStepLR",
        "kwargs": {
            "milestones": [10, 150, 250, 350],
            "gamma": 0.1,
        },
    },
    "grad_clip": 1.0,
    "use_mixup": True,
    "codes_noise_level": 0.1,
    "similarities_reg_coef": 15.0,  # Multiplicative factor applied on logits.
    "ce_loss_scale": 1.0,  # Scale factor applied on ce loss.
    "bin_loss_scale": 10.0,  # Scale factor applied on bin loss.
    "train_classification_head": False,
    "train_base_model": False,
    "n_workers": 4,
    "attacker_config": None,
    # "attacker_config": ATTACKER_CONFIG_CIFAR10,
    "projection_type": "very_large",
    "use_mixed_precision": False,
    "pretrained_path": "resources/cifar_model.p",
}

EXP_GROUPS = {k: [] for k in BASE_CONFIGS.keys()}

for dataset in EXP_GROUPS.keys():
    EXP_GROUPS[dataset].append(BASE_CONFIGS[dataset])

import torch


def l1_loss(logits, code_labels):
    """L1 loss"""
    loss = torch.nn.L1Loss(reduction="mean")(logits, code_labels)
    return loss

def l1_ce_loss(logits, codes_matrix, labels_one_hot, sim_coef):
    """Multi-code loss based on negative l1 distances"""

    repeated_logits = logits.unsqueeze(1).repeat(1, codes_matrix.size(0), 1)
    repeated_codes_matrix = codes_matrix.unsqueeze(0).repeat(logits.size(0), 1, 1)
    pointwise_neg_normalized_L1_distances = (
        # We normalize the L1 distances by dividing by the code length.
        # This is so we have some control over the range of the logits given to the loss via the temperature.
        -(repeated_logits - repeated_codes_matrix).abs().sum(-1)
        / codes_matrix.size(1)
    )

    if sim_coef is not None:
        pointwise_neg_normalized_L1_distances = sim_coef(
            pointwise_neg_normalized_L1_distances
        )

    loss = (
        -(pointwise_neg_normalized_L1_distances.log_softmax(dim=-1) * labels_one_hot)
        .sum(dim=-1)
        .mean()
    )

    return loss


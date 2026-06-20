"""Class-conditional subject alignment (CCSA) objective."""

import torch


def ccsa_loss(z, y, subjects):
    """Align subject centroids within each class in the embedding space.

    The assignment failure mode is subject-specific clustering despite reasonable
    class separability. This loss therefore aligns each subject/class centroid to
    the global centroid of the same class, rather than aligning all samples blindly
    and risking collapse of the motor-imagery classes.
    """
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    losses = []
    for c in torch.unique(y):
        class_mask = y == c
        if class_mask.sum() < 2:
            continue
        z_c = z[class_mask]
        s_c = subjects[class_mask]
        mu_c = z_c.mean(dim=0)
        for s in torch.unique(s_c):
            subject_mask = s_c == s
            if subject_mask.sum() < 1:
                continue
            mu_sc = z_c[subject_mask].mean(dim=0)
            losses.append(torch.sum((mu_sc - mu_c) ** 2))
    if len(losses) == 0:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()

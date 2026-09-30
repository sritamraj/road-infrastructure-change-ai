import torch

from src.segmentation.losses import DiceBCELoss


def test_dice_bce_loss_is_finite():
    loss_fn = DiceBCELoss()
    logits = torch.randn(1, 1, 32, 32)
    target = torch.randint(0, 2, (1, 1, 32, 32)).float()
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

import torch

from src.change_detection.models import SiameseChangeNet
from src.segmentation.models import UNet, build_model


def test_unet_output_shape():
    model = UNet().eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 1, 64, 64)


def test_siamese_output_shape():
    model = SiameseChangeNet().eval()
    image_a = torch.randn(1, 3, 64, 64)
    image_b = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model(image_a, image_b)
    assert output.shape == (1, 1, 64, 64)


def test_build_model_returns_unet():
    model = build_model()
    assert isinstance(model, UNet)

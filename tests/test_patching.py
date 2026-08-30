import torch

from surrogate3d.models.patching import patchify3d, unpatchify3d


def test_patchify3d_roundtrip():
    x = torch.randn(2, 6, 16, 8, 24)
    patch_size = (8, 4, 8)
    patches = patchify3d(x, patch_size)
    restored = unpatchify3d(patches, channels=6, patch_size=patch_size)
    assert torch.equal(restored, x)

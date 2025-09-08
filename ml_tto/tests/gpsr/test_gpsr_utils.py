import pytest
from ml_tto.gpsr.utils import image_snr
import numpy as np


class TestGPSRUtils:
    @pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
    def test_image_snr(self, scale):
        img = np.zeros((100, 100))
        img[30:70, 30:70] = 1.0 * scale
        img[45:55, 45:55] = 2.5 * scale
        img[48:52, 48:52] = 5.0 * scale
        img += np.random.normal(0, 1.0, size=img.shape)

        # add a few random hot pixels
        for _ in range(20):
            x = np.random.randint(0, img.shape[0])
            y = np.random.randint(0, img.shape[1])
            img[x, y] += 50.0

        old_img = np.copy(img)

        snr = image_snr(img)

        if scale < 1.0:
            assert snr < 3.0
        elif scale == 1.0:
            assert np.allclose(snr, 3.0, atol=1.0)
        else:
            assert np.allclose(snr, 26.0, atol=2.0)

        # make sure it doesn't alter the original image
        assert np.allclose(img, old_img)

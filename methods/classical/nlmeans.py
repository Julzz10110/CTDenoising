import numpy as np
from skimage._shared import utils
from skimage._shared.utils import convert_to_float
from skimage.restoration._nl_means_denoising import (
    _nl_means_denoising_2d,
    _fast_nl_means_denoising_2d,
)

def non_local_means(image, patch_size=7, patch_distance=11, h=0.1, fast_mode=True, sigma=0.0,
    *,
    preserve_range=False
):
    ndim_no_channel = image.ndim - 1
    if ndim_no_channel != 2:
        raise NotImplementedError("Метод локализованного усреднения реализован для двумерных двухканальных изображений.\n")

    image = convert_to_float(image, preserve_range)
    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)

    kwargs = dict(s=patch_size, d=patch_distance, h=h, var=sigma * sigma)
    if ndim_no_channel == 2:
        nlm_func = _fast_nl_means_denoising_2d if fast_mode else _nl_means_denoising_2d
    dn = np.asarray(nlm_func(image, **kwargs))
    return dn
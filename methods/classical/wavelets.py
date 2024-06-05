import numpy as np
import numbers
import scipy.stats
from skimage.util.dtype import img_as_float
import skimage.color as color
from skimage.util import random_noise
from skimage.color.colorconv import ycbcr_from_rgb
from warnings import warn


def __sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]

    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("В настоящее время поддерживается только оценка гауссовского шума.")
    return sigma


def __scale_sigma_and_img_consistently(img, sigma, multichannel, rescale_sigma):
    if multichannel:
        if isinstance(sigma, numbers.Number) or sigma is None:
            sigma = [sigma] * img.shape[-1]
        elif len(sigma) != img.shape[-1]:
            raise ValueError(
                "Если сhannel_axis не имеет значения None, параметр sigma должна быть скаляром или иметь "
                "длина равна количеству каналов"
            )
    if img.dtype.kind != 'f':
        if rescale_sigma:
            range_pre = img.max() - img.min()
        img = img_as_float(img)
        if rescale_sigma:
            range_post = img.max() - img.min()
            # apply the same magnitude scaling to sigma
            scale_factor = range_post / range_pre
            if multichannel:
                sigma = [s * scale_factor if s is not None else s for s in sigma]
            elif sigma is not None:
                sigma *= scale_factor
    elif img.dtype == np.float16:
        img = img.astype(np.float32)
    return img, sigma


def _rescale_sigma_rgb2ycbcr(sigmas):
    if sigmas[0] is None:
        return sigmas
    sigmas = np.asarray(sigmas)
    rgv_variances = sigmas * sigmas
    for i in range(3):
        scalars = ycbcr_from_rgb[i, :]
        var_channel = np.sum(scalars * scalars * rgv_variances)
        sigmas[i] = np.sqrt(var_channel)
    return sigmas


def _bayes_thresh(details, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(details * details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh

def _universal_thresh(img, sigma):
    """Universal threshold used by the VisuShrink method"""
    return sigma * np.sqrt(2 * np.log(img.size))


def __wavelet_threshold(
    img,
    wavelet,
    method=None,
    threshold=None,
    sigma=None,
    mode='soft',
    wavelet_levels=None,
):  
    try:
        import pywt
    except ImportError:
        raise ImportError(
            'PyWavelets is not installed. Please ensure it is installed in '
            'order to use this function.'
        )

    wavelet = pywt.Wavelet(wavelet)
    if not wavelet.orthogonal:
        warn(
            f'Wavelet thresholding was designed for '
            f'use with orthogonal wavelets. For nonorthogonal '
            f'wavelets such as {wavelet.name},results are '
            f'likely to be suboptimal.'
        )

    original_extent = tuple(slice(s) for s in img.shape)

    if wavelet_levels is None:
        wavelet_levels = pywt.dwtn_max_level(img.shape, wavelet)
        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedecn(img, wavelet=wavelet, level=wavelet_levels)
    dcoeffs = coeffs[1:]

    if sigma is None:
        detail_coeffs = dcoeffs[-1]['d' * img.ndim]
        sigma = __sigma_est_dwt(detail_coeffs, distribution='Gaussian')

    if method is not None and threshold is not None:
        warn(
            f'Thresholding method {method} selected. The '
            f'user-specified threshold will be ignored.'
        )

    if threshold is None:
        var = sigma**2
        if method is None:
            raise ValueError("If method is None, a threshold must be provided.")
        elif method == "BayesShrink":
            # The BayesShrink thresholds from [1]_ in docstring
            threshold = [
                {key: _bayes_thresh(level[key], var) for key in level}
                for level in dcoeffs
            ]
        elif method == "VisuShrink":
            # The VisuShrink thresholds from [2]_ in docstring
            threshold = _universal_thresh(img, sigma)
        else:
            raise ValueError(f'Unrecognized method: {method}')

    if np.isscalar(threshold):
        # A single threshold for all coefficient arrays
        denoised_detail = [
            {
                key: pywt.threshold(level[key], value=threshold, mode=mode)
                for key in level
            }
            for level in dcoeffs
        ]
    else:
        # Dict of unique threshold coefficients for each detail coeff. array
        denoised_detail = [
            {
                key: pywt.threshold(level[key], value=thresh[key], mode=mode)
                for key in level
            }
            for thresh, level in zip(threshold, dcoeffs)
        ]
    denoised_coeffs = [coeffs[0]] + denoised_detail
    out = pywt.waverecn(denoised_coeffs, wavelet)[original_extent]
    out = out.astype(img.dtype)
    return out


def denoise_wavelet(
    img,
    sigma=None,
    wavelet='db1',
    mode='soft',
    wavelet_levels=None,
    convert2ycbcr=False,
    method='BayesShrink',
    rescale_sigma=True,
    *,
    channel_axis=None,
):
    multichannel = channel_axis is not None
    if method not in ["BayesShrink", "VisuShrink"]:
        raise ValueError(
            f'Недоступный метод: {method}. поддерживаются только '
            f'методы "BayesShrink" и "VisuShrink".'
        )

    # floating-point inputs are not rescaled, so don't clip their output.
    clip_output = img.dtype.kind != 'f'

    img, sigma = __scale_sigma_and_img_consistently(
        img, sigma, multichannel, rescale_sigma
    )
    
    if multichannel:
        if convert2ycbcr:
            out = color.rgb2ycbcr(img)
            # convert user-supplied sigmas to the new colorspace as well
            if rescale_sigma:
                sigma = _rescale_sigma_rgb2ycbcr(sigma)
            for i in range(3):
                # renormalizing this color channel to live in [0, 1]
                _min, _max = out[..., i].min(), out[..., i].max()
                scale_factor = _max - _min
                if scale_factor == 0:
                    # skip any channel containing only zeros!
                    continue
                channel = out[..., i] - _min
                channel /= scale_factor
                sigma_channel = sigma[i]
                if sigma_channel is not None:
                    sigma_channel /= scale_factor
                out[..., i] = denoise_wavelet(
                    channel,
                    wavelet=wavelet,
                    method=method,
                    sigma=sigma_channel,
                    mode=mode,
                    wavelet_levels=wavelet_levels,
                    rescale_sigma=rescale_sigma,
                )
                out[..., i] = out[..., i] * scale_factor
                out[..., i] += _min
            out = color.ycbcr2rgb(out)
        else:
            out = np.empty_like(img)
            for c in range(img.shape[-1]):
                out[..., c] = __wavelet_threshold(
                    img[..., c],
                    wavelet=wavelet,
                    method=method,
                    sigma=sigma[c],
                    mode=mode,
                    wavelet_levels=wavelet_levels,
                )
    else:
        out = __wavelet_threshold(
            img,
            wavelet=wavelet,
            method=method,
            sigma=sigma,
            mode=mode,
            wavelet_levels=wavelet_levels,
        )

    if clip_output:
        clip_range = (-1, 1) if img.min() < 0 else (0, 1)
        out = np.clip(out, *clip_range, out=out)
    return out

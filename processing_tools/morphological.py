
import numpy as np

def dilation(img, selem=None, out=None):
    dim = img.ndim
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if not img.dtype == np.bool:
        img = img.astype(np.bool)
    if selem is None:
        if dim == 1:
            selem = np.ones(shape=[3], dtype=np.bool)
        elif dim == 2:
            selem = np.zeros(shape=[3, 3], dtype=np.bool)
            selem[1, :] = True
            selem[:, 1] = True
        elif dim == 3:
            selem = np.zeros(shape=[3, 3, 3], dtype=np.bool)
            selem[:, 1, 1] = True
            selem[1, :, 1] = True
            selem[1, 1, :] = True
    else:
        if not isinstance(selem, np.ndarray):
            selem = np.asarray(selem, dtype=np.bool)
        if not selem.dtype == np.bool:
            selem = selem.astype(np.bool)
        if any([num_pixels % 2 == 0 for num_pixels in selem.shape]):
            raise ValueError('Поддерживаются только элементы структуры нечетного размера в каждом направлении.')
    perimeter_img = _get_perimeter_image(img)
    perimeter_coords = np.where(perimeter_img)
    if out is None:
        return_out = True
        out = img.copy()
    else:
        return_out = False
        out[:] = img[:]

    if dim == 1:
        sx = selem.shape[0]
        rx = sx // 2
        lx = img.shape[0]
        for ix in perimeter_coords[0]:
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            out[jx_b:jx_e] |= selem[kx_b:kx_e]

    if dim == 2:
        rx, ry = [n // 2 for n in selem.shape]
        lx = img.shape
        sx, sy = selem.shape
        lx, ly = img.shape
        for ix, iy in zip(perimeter_coords[0], perimeter_coords[1]):
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            (jy_b, jy_e), (ky_b, ky_e) = _generate_array_indices(iy, ry, sy, ly)
            out[jx_b:jx_e, jy_b:jy_e] |= selem[kx_b:kx_e, ky_b:ky_e]

    if dim == 3:
        rx, ry, rz = [n // 2 for n in selem.shape]
        sx, sy, sz = selem.shape
        lx, ly, lz = img.shape
        for ix, iy, iz in zip(perimeter_coords[0], perimeter_coords[1], perimeter_coords[2]):
            (jx_b, jx_e), (kx_b, kx_e) = _generate_array_indices(ix, rx, sx, lx)
            (jy_b, jy_e), (ky_b, ky_e) = _generate_array_indices(iy, ry, sy, ly)
            (jz_b, jz_e), (kz_b, kz_e) = _generate_array_indices(iz, rz, sz, lz)
            out[jx_b:jx_e, jy_b:jy_e, jz_b:jz_e] |= selem[kx_b:kx_e, ky_b:ky_e, kz_b:kz_e]

    if return_out:
        return out

def erosion(img, selem=None, out=None):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if not img.dtype == np.bool:
        img = img.astype(np.bool)

    out_img = dilation(~img, selem, out)

    if out is None:
        return ~out_img
    else:
        out[:] = ~out[:]

def closing(img, selem=None, out=None):
    out_img = erosion(dilation(img, selem), selem, out)
    if out is None:
        return out_img

def opening(image, selem=None, out=None):
    out_image = dilation(erosion(image, selem), selem, out)
    if out is None:
        return out_image

def _get_perimeter_image(img):
    dim = img.ndim
    if dim > 3:
        raise RuntimeError('Двоичное изображение в формате 3D или выше не поддерживается.')
    count = np.zeros_like(img, dtype=np.uint8)
    inner = np.zeros_like(img, dtype=np.bool)

    count[1:] += img[:-1]
    count[:-1] += img[1:]

    if dim == 1:
        inner |= img == 2
        for i in [0, -1]:
            inner[i] |= count[i] == 1
        return img & (~inner)

    count[:, 1:] += img[:, :-1]
    count[:, :-1] += img[:, 1:]
    if dim == 2:
        inner |= count == 4
        for i in [0, -1]:
            inner[i] |= count[i] == 3
            inner[:, i] |= count[:, i] == 3
        for i in [0, -1]:
            for j in [0, -1]:
                inner[i, j] |= count[i, j] == 2
        return img & (~inner)

    count[:, :, 1:] += img[:, :, :-1]
    count[:, :, :-1] += img[:, :, 1:]

    if dim == 3:
        inner |= count == 6
        for i in [0, -1]:
            inner[i] |= count[i] == 5
            inner[:, i] |= count[:, i] == 5
            inner[:, :, i] |= count[:, :, i] == 5
        for i in [0, -1]:
            for j in [0, -1]:
                inner[i, j] |= count[i, j] == 4
                inner[:, i, j] |= count[:, i, j] == 4
                inner[:, i, j] |= count[:, i, j] == 4
                inner[i, :, j] |= count[i, :, j] == 4
                inner[i, :, j] |= count[i, :, j] == 4
        for i in [0, -1]:
            for j in [0, -1]:
                for k in [0, -1]:
                    inner[i, j, k] |= count[i, j, k] == 3
        return img & (~inner)
    raise RuntimeError('Ошибка исполнения.')

def _generate_array_indices(selem_center, selem_radius, selem_length, result_length):
    result_begin = selem_center - selem_radius
    result_end = selem_center + selem_radius + 1
    selem_begin = -result_begin if result_begin < 0 else 0
    result_begin = max(0, result_begin)
    selem_end = selem_length -(result_end - result_length) \
                    if result_end > result_length else selem_length
    return (result_begin, result_end), (selem_begin, selem_end)
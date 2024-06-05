import numpy as np

def add_gauss_noise(img, sigma):
    gauss_noise = np.random.normal(0, sigma, img.shape)
    np.add(img, gauss_noise, out=img, casting="unsafe")
    return img

def add_poisson_noise(img, photon_count):
    opt = dict(dtype=np.float32)
    img = np.exp(-img, **opt)
    img = np.random.poisson(img * photon_count)
    img[img == 0] = 1
    img = img / photon_count
    img = -np.log(img, **opt)
    return img

def add_saltpepper_noise(img, probability):
    noise = np.random.rand(img.shape[0], img.shape[1])
    img[noise < probability] = 0
    img[noise > (1-probability)] = 255
    return img


def transmittance(sinogram):
    return np.mean(np.exp(-sinogram)[sinogram > 0])


def absorption(sinogram):
    return 1 - transmittance(sinogram)

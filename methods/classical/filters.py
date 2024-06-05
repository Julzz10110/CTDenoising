import cv2
import numpy as np
from itertools import product
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros

def gaussian(x, sigma):
    return (1.0/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return np.sqrt(np.abs((x1-x2)**2-(y1-y2)**2))

def bilateral_filter(img, diameter, sigma_i, sigma_s):
    out_img = np.zeros(img.shape)

    for row in range(len(img)):
        for col in range(len(img[0])):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x =row - (diameter/2 - k)
                    n_y =col - (diameter/2 - l)
                    if n_x >= len(img):
                        n_x -= len(img)
                    if n_y >= len(img[0]):
                        n_y -= len(img[0])
                    gi = gaussian(img[int(n_x)][int(n_y)] - img[row][col], sigma_i)
                    gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi * gs
                    filtered_image = (filtered_image) + (img[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            out_img[row][col] = int(np.round(filtered_image))
    return out_img

def gen_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
    return g

def gaussian_filter(img, sigma, k_size):
    height, width = img.shape[0], img.shape[1]
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1
    image_array = zeros((dst_height * dst_width, k_size * k_size))
    for row, (i, j) in enumerate(product(range(dst_height), range(dst_width))):
        window = ravel(img[i : i + k_size, j : j + k_size])
        image_array[row, :] = window
    gaussian_kernel = gen_gaussian_kernel(k_size, sigma)
    filter_array = ravel(gaussian_kernel)
    out_img = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)
    return out_img

def median_filter(img, kernel_size):
    temp = []
    indexer = kernel_size // 2
    out_img = np.zeros((len(img), len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[0])):
            for z in range(kernel_size):
                if i + z - indexer < 0 or i + z - indexer > len(img) - 1:
                    for c in range(kernel_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(img[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(kernel_size):
                            temp.append(img[i + z - indexer][j + k - indexer])
            temp.sort()
            out_img[i][j] = temp[len(temp) // 2]
            temp = []
    return out_img
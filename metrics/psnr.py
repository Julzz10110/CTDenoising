import numpy as np
import math

# Вычисление PSNR
def calculate_psnr(img1, img2):
    # Значения пикселей изображений img1 и img2 в диапазоне [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 *abs(math.log10(255.0 / math.sqrt(mse)))
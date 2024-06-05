#from methods.classical.bm3d import BM3D
#from methods.classical.nlmeans import non_local_means
#from methods.classical.wavelets import denoise_wavelet
from metrics.psnr import calculate_psnr
#from skimage.util import random_noise
from train import train_model
from evaluate import evaluate_model

import matplotlib.pyplot as plt

import cv2


if __name__ == '__main__':
    img = cv2.imread('input/rec_00600.tif')
    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    '''
    sigma = 25 # variance of the noise
    
    lamb2d = 2.0
    
    lamb3d = 2.7
    
    thre_dist_1 = 2500 # threshold distance
    
    max_match_1 = 16 # max matched blocks
    
    block_size_1 = 8
    
    spdup_factor_1 = 3 # pixel jump for new reference block
    
    window_size_1 = 39 # search window size  
    
    thre_dist_2 = 400
    
    max_match_2 = 32
    
    block_size_2 = 8
    
    spdup_factor_2 = 3
    
    window_size_2 = 39

    kaiser_window_beta = 2.0

    #processed_img = BM3D(img, sigma, lamb2d, lamb3d, thre_dist_1, max_match_1, block_size_1, spdup_factor_1, window_size_1, \
    #     thre_dist_2, max_match_2, block_size_2, spdup_factor_2, window_size_2, kaiser_window_beta)
    '''

    '''
    sigma=0.12
    noisy = random_noise(img, var=sigma**2)

    im_bayes = denoise_wavelet(
    noisy,
    channel_axis=-1,
    method='BayesShrink',
    mode='soft',
    rescale_sigma=True,
  )
    print("PSNR: ", calculate_psnr(img, im_bayes))
    '''

    '''
    sigma=0.12
    noisy = random_noise(img, var=sigma**2)
    denoised_img = non_local_means(img)
    cv2.imwrite('output/rec_00600_denoised_wavelet.tif', denoised_img)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5), sharex=True, sharey=True)
    plt.gray()

    ax[0, 0].imshow(img)
    ax[0, 0].axis('off')
    ax[0, 0].set_title(f'Зашумленное изображение\nPSNR={calculate_psnr(img, noisy):0.6g}')
    ax[0, 1].imshow(denoised_img)
    ax[0, 1].axis('off')
    ax[0, 1].set_title(f'NLM-шумоподавление\n(BayesShrink)\nPSNR={calculate_psnr(img, denoised_img):0.6g}')


    plt.show()
    '''
    #denoised_img = non_local_means(img)
    #cv2.imwrite('output/rec_00600_denoised_nlm.png', denoised_img)
    train_model(train_dir='input/beton/', output_dir='output/weights_1/', network='dncnn', num_splits=2, \
                strategy="X:1", epochs=600, batch_size=16, data_scaling=200)
    evaluate_model(input_dir='input/beton/', weights_path='output/weights_1/weights.torch', output_dir='output/', num_splits = 2, \
                   strategy = "X:1", batch_size = 16, data_scaling=200, network='dncnn')

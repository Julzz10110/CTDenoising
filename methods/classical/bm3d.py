import os
import cv2
import time
import sys
from scipy.fftpack import dct, idct
from processing_tools.noise import add_gauss_noise
from metrics.psnr import calculate_psnr
import numpy as np


def __initialization(img, block_size, kaiser_window_beta):
    init_img = np.zeros(img.shape, dtype=float)
    init_weight = np.zeros(img.shape, dtype=float)
    window = np.matrix(np.kaiser(block_size, kaiser_window_beta))
    init_kaiser = np.array(window.T * window)            

    return init_img, init_weight, init_kaiser


def __search_window(img, ref_point, block_size, window_size):
    if block_size >= window_size:
        print('ОШИБКА: block_size меньше чем window_size.\n')
        exit()
    margin = np.zeros((2,2), dtype = int)
    margin[0, 0] = max(0, ref_point[0]+int((block_size-window_size)/2))
    margin[0, 1] = max(0, ref_point[1]+int((block_size-window_size)/2))            
    margin[1, 0] = margin[0, 0] + window_size
    margin[1, 1] = margin[0, 1] + window_size            
    if margin[1, 0] >= img.shape[0]:
        margin[1, 0] = img.shape[0] - 1
        margin[0, 0] = margin[1, 0] - window_size
    if margin[1, 1] >= img.shape[1]:
        margin[1, 1] = img.shape[1] - 1
        margin[0, 1] = margin[1, 1] - window_size
    return margin


def __dct2D(A):
    return dct(dct(A, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')


def __idct2D(A):
    return idct(idct(A, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho') 

    
def __pre_DCT(img, block_size):
    block_DCT_all = np.zeros((img.shape[0]-block_size, img.shape[1]-block_size, block_size, block_size),\
                            dtype = float)
    for i in range(block_DCT_all.shape[0]):
        for j in range(block_DCT_all.shape[1]):
            block = img[i:i+block_size, j:j+block_size]
            block_DCT_all[i, j, :, :] = __dct2D(block.astype(np.float64))      
    return block_DCT_all


def __grouping_1(noisy_img, ref_point, block_DCT_all, block_size, thre_dist, max_match, window_size, sigma, lamb2d):
    window_loc = __search_window(noisy_img, ref_point, block_size, window_size)
    block_num_searched = (window_size-block_size+1)**2                   
    block_pos = np.zeros((block_num_searched, 2), dtype = int)
    block_group = np.zeros((block_num_searched, block_size, block_size), dtype = float)
    dist_m = np.zeros(block_num_searched, dtype = float)
    ref_DCT = block_DCT_all[ref_point[0], ref_point[1], :, :]
    match_cnt = 0

    for i in range(window_size-block_size+1):
        for j in range(window_size-block_size+1):
            searched_DCT = block_DCT_all[window_loc[0, 0]+i, window_loc[0, 1]+j, :, :]
            dist = __calculate_dist_1(ref_DCT, searched_DCT, sigma, lamb2d)
            if dist < thre_dist:
                block_pos[match_cnt, :] = [window_loc[0, 0]+i, window_loc[0, 1]+j]
                block_group[match_cnt, :, :] = searched_DCT
                dist_m[match_cnt] = dist
                match_cnt += 1   
    if match_cnt <= max_match:    
        block_pos = block_pos[:match_cnt, :]
        block_group = block_group[:match_cnt, :, :]
    
    else:
        idx = np.argpartition(dist_m[:match_cnt], max_match)
        block_pos = block_pos[idx[:max_match], :]
        block_group = block_group[idx[:max_match], :]

    return block_pos, block_group


def __calculate_dist_1(block_DCT_1, block_DCT_2, sigma, lamb2d):
    if block_DCT_1.shape != block_DCT_1.shape:
        print('ОШИБКА: размерность двух блоков DCT неодинакова.\n')
        sys.exit()
        
    elif block_DCT_1.shape[0] != block_DCT_1.shape[1]:
        print('ОШИБКА: блок DCT не является квадратной матрицей.\n')
        sys.exit()
    
    block_size = block_DCT_1.shape[0]
    
    if sigma > 40:

        thre_value = lamb2d * sigma

        block_DCT_1 = np.where(abs(block_DCT_1) < thre_value, 0, block_DCT_1)

        block_DCT_2 = np.where(abs(block_DCT_2) < thre_value, 0, block_DCT_2)

    return np.linalg.norm(block_DCT_1 - block_DCT_2)**2 / (block_size**2)


def __filtering3d_1(block_group, sigma, lamb3d):
    thre_value = lamb3d * sigma
    nonzero_cnt = 0
    
    for i in range(block_group.shape[1]):
        for j in range(block_group.shape[2]):
            third_vector = dct(block_group[:, i, j], norm = 'ortho') # 1D DCT
            third_vector[abs(third_vector[:]) < thre_value] = 0.
            nonzero_cnt += np.nonzero(third_vector)[0].size
            block_group[:, i, j] = list(idct(third_vector, norm = 'ortho'))

    return block_group, nonzero_cnt


def __aggregation_1(sigma, block_group, block_pos, basic_img, basic_weight, basic_kaiser, nonzero_cnt):
    if nonzero_cnt < 1:
        block_weight = 1.0 * basic_kaiser
    else:
        block_weight = (1./(sigma**2 * nonzero_cnt)) * basic_kaiser

    for i in range(block_pos.shape[0]):
        basic_img[block_pos[i, 0]:block_pos[i, 0]+block_group.shape[1],\
                 block_pos[i, 1]:block_pos[i, 1]+block_group.shape[2]]\
                                 += block_weight * __idct2D(block_group[i, :, :])
        basic_weight[block_pos[i, 0]:block_pos[i, 0]+block_group.shape[1],\
                    block_pos[i, 1]:block_pos[i, 1]+block_group.shape[2]] += block_weight
        
    return basic_img, basic_weight


def __BM3D_1(noisy_img, sigma, lamb2d, lamb3d, block_size, thre_dist, max_match, window_size, spdup_factor, kaiser_window_beta):

    basic_img, basic_weight, basic_kaiser = __initialization(noisy_img, block_size, kaiser_window_beta)
    block_DCT_all = __pre_DCT(noisy_img, block_size)

    for i in range(int((noisy_img.shape[0]-block_size)/spdup_factor)+2):

        for j in range(int((noisy_img.shape[1]-block_size)/spdup_factor)+2):

            ref_point = [min(spdup_factor*i, noisy_img.shape[0]-block_size-1), \
                        min(spdup_factor*j, noisy_img.shape[1]-block_size-1)]

            block_pos, block_group = __grouping_1(noisy_img, ref_point, block_DCT_all, block_size, \
                                                  thre_dist, max_match, window_size, sigma, lamb2d)
            
            block_group, nonzero_cnt = __filtering3d_1(block_group, sigma, lamb3d)

            basic_img, basic_weight = __aggregation_1(sigma, block_group, block_pos, basic_img, basic_weight, basic_kaiser, nonzero_cnt)

    basic_weight = np.where(basic_weight == 0, 1, basic_weight)
    
    basic_img[:, :] /= basic_weight[:, :]

#    basicImg = (np.matrix(basicImg, dtype=int)).astype(np.uint8)

    return basic_img


    
def __grouping_2(basic_img, noisy_img, ref_point, block_size, thre_dist, max_match, window_size, sigma, lamb2d,
                   block_DCT_basic, block_DCT_noisy):
    
    window_loc = __search_window(basic_img, ref_point, block_size, window_size)
    
    block_num_searched = (window_size-block_size+1)**2
                         
    block_pos = np.zeros((block_num_searched, 2), dtype = int)

    block_group_basic = np.zeros((block_num_searched, block_size, block_size), dtype = float)

    block_group_noisy = np.zeros((block_num_searched, block_size, block_size), dtype = float)

    dist_m = np.zeros(block_num_searched, dtype = float)
    
    match_cnt = 0

    for i in range(window_size-block_size+1):

        for j in range(window_size-block_size+1):

            searched_point = [window_loc[0, 0]+i, window_loc[0, 1]+j]

            dist = __calculate_dist_2(basic_img, ref_point, searched_point, block_size)

            if dist < thre_dist:
                
                block_pos[match_cnt, :] = searched_point

                dist_m[match_cnt] = dist

                match_cnt += 1
         
    if match_cnt <= max_match:
        block_pos = block_pos[:match_cnt, :]
    
    else:
        idx = np.argpartition(dist_m[:match_cnt], max_match)
        block_pos = block_pos[idx[:max_match], :]
        
    for i in range(block_pos.shape[0]):
        
        similar_point = block_pos[i, :]
        
        block_group_basic[i, :, :] = block_DCT_basic[similar_point[0], similar_point[1], :, :]
        
        block_group_noisy[i, :, :] = block_DCT_noisy[similar_point[0], similar_point[1], :, :]
        
    block_group_basic = block_group_basic[:block_pos.shape[0], :, :]
    
    block_group_noisy = block_group_noisy[:block_pos.shape[0], :, :]

    return block_pos, block_group_basic, block_group_noisy
    

def __calculate_dist_2(img, point_1, point_2, block_size):
    
    block_1 = (img[point_1[0]:point_1[0]+block_size, point_1[1]:point_1[1]+block_size]).astype(np.float64)
    
    block_2 = (img[point_2[0]:point_2[0]+block_size, point_2[1]:point_2[1]+block_size]).astype(np.float64)
    
    return np.linalg.norm(block_1-block_2)**2 / (block_size**2)


def __filtering3d_2(block_group_basic, block_group_noisy, sigma, lamb3d):
    
    weight = 0
    
    coef = 1.0 / block_group_noisy.shape[0]
    
    for i in range(block_group_noisy.shape[1]):
        
        for j in range(block_group_noisy.shape[2]):
            
            vec_basic = dct(block_group_basic[:, i, j], norm = 'ortho')
            
            vec_noisy = dct(block_group_noisy[:, i, j], norm = 'ortho')
            
            vec_value = vec_basic**2 * coef
            
            vec_value /= (vec_value + sigma**2)
            
            vec_noisy *= vec_value
            
            weight += np.sum(vec_value)
#            for k in range(BlockGroup_noisy.shape[0]):
#                
#                Value = Vec_basic[k]**2 * coef
#                
#                Value /= (Value + sigma**2) # pixel weight 
#                
#                Vec_noisy[k] = Vec_noisy[k] * Value
#                
#                Weight += Value
            
            block_group_noisy[:, i, j] = list(idct(vec_noisy, norm = 'ortho'))
    
    if weight > 0:
    
        wiener_weight = 1./(sigma**2 * weight)
    
    else:
        
        wiener_weight = 1.0
                
    return block_group_noisy, wiener_weight


def __aggregation_2(block_group_noisy, wiener_weight, block_pos, final_img, final_weight, final_kaiser):
    
    block_weight = wiener_weight * final_kaiser

    for i in range(block_pos.shape[0]):
        
        final_img[block_pos[i, 0]:block_pos[i, 0]+block_group_noisy.shape[1],\
                 block_pos[i, 1]:block_pos[i, 1]+block_group_noisy.shape[2]]\
                                 += block_weight * __idct2D(block_group_noisy[i, :, :])

        final_weight[block_pos[i, 0]:block_pos[i, 0]+block_group_noisy.shape[1],\
                    block_pos[i, 1]:block_pos[i, 1]+block_group_noisy.shape[2]] += block_weight
    
    return final_img, final_weight
    

def __BM3D_2(basic_img, noisy_img, sigma, lamb2d, lamb3d, block_size, thre_dist, max_match, window_size, spdup_factor, kaiser_window_beta):

    final_img, final_weight, final_kaiser = __initialization(basic_img, block_size, kaiser_window_beta)

    block_DCT_basic = __pre_DCT(basic_img, block_size)

    block_DCT_noisy = __pre_DCT(noisy_img, block_size)

    for i in range(int((basic_img.shape[0]-block_size)/spdup_factor)+2):

        for j in range(int((basic_img.shape[1]-block_size)/spdup_factor)+2):

            ref_point = [min(spdup_factor*i, basic_img.shape[0]-block_size-1), \
                        min(spdup_factor*j, basic_img.shape[1]-block_size-1)]

            block_pos, block_group_basic, block_group_noisy = __grouping_2(basic_img, noisy_img, \
                                                                          ref_point, block_size, \
                                                                          thre_dist, max_match, \
                                                                          window_size, \
                                                                          sigma, lamb2d, \
                                                                          block_DCT_basic, \
                                                                          block_DCT_noisy)

            block_group_noisy, wiener_weight = __filtering3d_2(block_group_basic, block_group_noisy, sigma, lamb3d)

            final_img, final_weight = __aggregation_2(block_group_noisy, wiener_weight, block_pos, final_img, final_weight, \
                              final_kaiser)
    
    final_weight = np.where(final_weight == 0, 1, final_weight)

    final_img[:, :] /= final_weight[:, :]

#   finalImg = (np.matrix(finalImg, dtype=int)).astype(np.uint8)

    return final_img




def BM3D(img, sigma, lamb2d, lamb3d, thre_dist_1, max_match_1, block_size_1, spdup_factor_1, window_size_1, 
         thre_dist_2, max_match_2, block_size_2, spdup_factor_2, window_size_2, kaiser_window_beta):

    cv2.setUseOptimized(True)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    noisy_img = add_gauss_noise(img, sigma)
    
    start_time = time.time()
    
    basic_img = __BM3D_1(noisy_img, sigma, lamb2d, lamb3d, block_size_1, thre_dist_1, max_match_1, window_size_1, spdup_factor_1, kaiser_window_beta)
    
    basic_PSNR = calculate_psnr(img, basic_img)
    
    #print('Значение PSNR исходного изображения равно {} дБ.\n'.format(basic_PSNR))
    
    basic_img_uint = np.zeros(img.shape)
    
    cv2.normalize(basic_img, basic_img_uint, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    
    basic_img_uint = basic_img_uint.astype(np.uint8)
    
   # cv2.imwrite('basicdog.png', basic_img_uint)
    '''
    if cv2.imwrite('basicdog.png', basic_img_uint) == True:
        
        print('Базовое изображение успешно сохранено.\n')
        
        step1_time = time.time()
    
        print('Время выполнения базовой оценки составляет', step1_time - start_time, 'секунд.\n')
        
    else:
        
        print('ОШИБКА: базовая оценка не реконструирована успешно.\n')
        
        sys.exit()
    '''
      
    final_img = __BM3D_2(basic_img, noisy_img, sigma, lamb2d, lamb3d, block_size_2, \
                       thre_dist_2, max_match_2, window_size_2, spdup_factor_2, kaiser_window_beta)
    
    final_PSNR = calculate_psnr(img, final_img)
    
    #print('Значение PSNR итогового изображения равно {} дБ.\n'.format(final_PSNR))
    
    cv2.normalize(final_img, final_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    
    final_img = final_img.astype(np.uint8)
    
    # cv2.imwrite('finaldog.png', final_img)

    '''
    if cv2.imwrite('finaldog.png', final_img) == True:
        print('Итоговое изображение успешно сохранено.\n')
        step2_time = time.time()
        print('Время выполнения итоговй оценки составляет', step2_time - step1_time, 'секунд.\n')
        
    else:
        print('ОШИБКА: итоговая оценка не реконструирована успешно.\n')
        sys.exit()
    '''
    return final_img
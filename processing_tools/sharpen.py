import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
  

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    sharpened_img = cv2.filter2D(img, -1, kernel) 
    return sharpened_img
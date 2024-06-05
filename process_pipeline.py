import json
import cv2
import numpy as np
import processing_tools.noise as noise
import processing_tools.blur as blur
import processing_tools.sharpen as sharpen
import processing_tools.brightness as brightness

import methods.classical.filters as filters
import methods.classical.bm3d as bm3d
import methods.classical.nlmeans as nlmeans

MODULES = [noise, blur, sharpen, brightness, filters, bm3d, nlmeans]

def execute(img, func_name, params):
    for module in MODULES:
        try:
            exec_func = getattr(module, func_name)
        except Exception:
            pass
    
    return np.array(exec_func(img, *params))


if __name__ == '__main__':
    with open('denoising_pipeline.json') as f:
        data = json.load(f)
    
    img = cv2.imread(data['input']['file'])
    #img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    if data['preprocessing']:
        for operation in data['preprocessing']['operations_queue']:
            operation_name, params = list(operation.keys())[0], list(operation.values())[0]
            if not operation_name:
                raise Exception("Пустое поле операции")
            print(operation_name, params)
            img = execute(img, operation_name, params)
            cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            cv2.imwrite('test_pre.png', img)
    
    if data['denoising']:
        stage_names = sorted(data['denoising'].keys())
        print(stage_names)
        for stage_name in stage_names:
            method_type = data['denoising'][stage_name]['method_type']
            if method_type == 'classical':
                method_name = data['denoising'][stage_name]['method_name']
                params = data['denoising'][stage_name]['params']
                print(method_name, params)
                img = execute(img, method_name, params)
                cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
                cv2.imwrite('test_denoised.png', img)

    if data['postprocessing']:
        print("test")
        for operation in data['postprocessing']['operations_queue']:
            operation_name, params = list(operation.keys())[0], list(operation.values())[0]
            if not operation_name:
                raise Exception("Пустое поле операции")
            print(operation_name, params)
            img = execute(img, operation_name, params)
            cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            cv2.imwrite('test_post.png', img)

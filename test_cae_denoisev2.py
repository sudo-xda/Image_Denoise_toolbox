import os.path
import logging
import argparse

import numpy as np
from datetime import datetime
from collections import OrderedDict

import tensorflow as tf

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def imread_uint(path, n_channels=1):
    img = tf.keras.utils.load_img(path, color_mode="grayscale" if n_channels == 1 else "rgb")
    return np.array(img, dtype=np.uint8)

def uint2single(img):
    return np.float32(img) / 255.0

def tensor2uint(img):
    img = np.clip(img * 255.0, 0, 255)
    return np.uint8(img)

def imsave(img, path):
    tf.keras.utils.save_img(path, img, scale=False)

def calculate_psnr(img1, img2, border=0):
    img1 = img1[border:-border, border:-border] if border != 0 else img1
    img2 = img2[border:-border, border:-border] if border != 0 else img2
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')

def calculate_ssim(img1, img2, border=0):
    from skimage.metrics import structural_similarity as ssim
    img1 = img1[border:-border, border:-border] if border != 0 else img1
    img2 = img2[border:-border, border:-border] if border != 0 else img2
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='CAE', help='dncnn_15, dncnn_25, dncnn_50, dncnn_gray_blind, dncnn_color_blind, dncnn3')
    parser.add_argument('--testset_name', type=str, default='ct1', help='test set, bsd68 | set12')
    parser.add_argument('--noise_level_img', type=int, default=55, help='noise level: 15, 25, 50')
    parser.add_argument('--show_img', type=bool, default=False, help='show the image')
    parser.add_argument('--model_pool', type=str, default='model_zoo', help='path of model_zoo')
    parser.add_argument('--testsets', type=str, default='testsets', help='path of testing folder')
    parser.add_argument('--results', type=str, default='results', help='path of results')
    parser.add_argument('--need_degradation', type=bool, default=True, help='add noise or not')
    args = parser.parse_args()

    if 'color' in args.model_name:
        n_channels = 3        # fixed, 1 for grayscale image, 3 for color image
    else:
        n_channels = 1        # fixed for grayscale image

    result_name = args.testset_name + '_' + args.model_name     # fixed
    model_path = os.path.join(args.model_pool, args.model_name+'.h5')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(args.testsets, args.testset_name) # L_path, for Low-quality images
    H_path = L_path                               # H_path, for High-quality images
    E_path = os.path.join(args.results, result_name)   # E_path, for Estimated images
    mkdir(E_path)

    logger_name = result_name
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False

    # ----------------------------------------
    # Load TensorFlow model
    # ----------------------------------------

    model = tf.keras.models.load_model(model_path, compile=False)
    logger.info('Model path: {:s}'.format(model_path))
    logger.info('Model loaded successfully')

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}, image sigma:{}'.format(args.model_name, args.noise_level_img))
    logger.info(L_path)
    L_paths = [os.path.join(L_path, f) for f in os.listdir(L_path) if os.path.isfile(os.path.join(L_path, f))]
    H_paths = L_paths if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = imread_uint(img, n_channels=n_channels)
        img_L = uint2single(img_L)

        if args.need_degradation:  # degradation process
            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, args.noise_level_img/255., img_L.shape)

        img_L = np.expand_dims(img_L, axis=0) if n_channels == 1 else np.expand_dims(img_L, axis=0)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        img_E = model.predict(img_L, verbose=0)[0]
        img_E = tensor2uint(img_E.squeeze())

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------

            img_H = imread_uint(H_paths[idx], n_channels=n_channels)

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            psnr = calculate_psnr(img_E, img_H)
            ssim = calculate_ssim(img_E, img_H)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

        # ------------------------------------
        # save results
        # ------------------------------------

        imsave(img_E, os.path.join(E_path, img_name+ext))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()

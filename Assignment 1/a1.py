#a1.py
#Assignment 1 (or 0, depending on who asks)
#by Abe Davis
#
#You can change the 'as' calls below if you want to call functions differently (i.e. numpy.function() instead of np.function())

import numpy as np


def brightness(im, factor):
    imb = im.copy()  # always work on a copy of the image!
    return imb * factor


def contrast(im, factor, midpoint=0.3):
    imc = im.copy()
    imc = imc * factor + midpoint * (1 - factor)
    return imc


def frame(im):
    imf = im.copy()
    shape = imf.shape
    imf[0, :, :] = 0
    imf[:, 0, :] = 0
    imf[shape[0]-1, :, :] = 0
    imf[:, shape[1]-1, :] = 0
    return imf


def BW(im, weights=[0.3, 0.6, 0.1]):
    img = im.copy()
    greyVal = np.dot(img[:, :], weights)
    img[:, :, 0] = greyVal
    img[:, :, 1] = greyVal
    img[:, :, 2] = greyVal
    return img


def lumiChromi(im):
    imL = BW(im)
    imC = im.copy() / imL
    return (imL, imC)


def brightnessContrastLumi(im, brightF, contrastF, midpoint=0.3):
    (imL, imC) = lumiChromi(im)
    imL = brightness(imL, brightF)
    imL = contrast(imL, contrastF, midpoint)
    return imL * imC


def rgb2yuv(im):
    imyuv = im.copy()
    transM = np.array([0.299, 0.587, 0.114, -0.14713, -0.28886, 0.436, 0.615, -0.51499, -0.10001]).reshape((3,3))
    for row_index, col_index in np.ndindex(imyuv.shape[0:2]):
        imyuv[row_index, col_index, :] = np.dot(transM, imyuv[row_index, col_index, :])
    return imyuv


def yuv2rgb(im):
    imrgb = im.copy()
    transM = np.array([[1, 0, 1.13983, 1, -0.39465, -0.58060, 1, 2.03211, 0]]).reshape((3, 3))
    for row_index, col_index in np.ndindex(imrgb.shape[0:2]):
        imrgb[row_index, col_index, :] = np.dot(transM, imrgb[row_index, col_index, :])
    return imrgb


def saturate(im, k):
    imS = rgb2yuv(im)
    for row_index, col_index in np.ndindex(imS.shape[0:2]):
        imS[row_index, col_index, 1] = imS[row_index, col_index, 1]*k
        imS[row_index, col_index, 2] = imS[row_index, col_index, 2]*k
    return yuv2rgb(imS)


#spanish castle URL: http://www.johnsadowski.com/big_spanish_castle.php
#HINT: to invert color for a YUV image, negate U and V
def spanish(im):
    imC = rgb2yuv(im)
    for row_index, col_index in np.ndindex(imC.shape[0:2]):
        imC[row_index, col_index, 0] = 0.4
        imC[row_index, col_index, 1] = -1 * imC[row_index, col_index, 1]
        imC[row_index, col_index, 2] = -1 * imC[row_index, col_index, 2]
    imC = yuv2rgb(imC)
    imL = BW(im)
    xcenter = im.shape[0] / 2
    ycenter = im.shape[1] / 2
    imC[xcenter, ycenter, :] = 0
    imL[xcenter, ycenter, :] = 0
    return (imL, imC)


def histogram(im, N):
    hist = np.zeros(N)
    imL = BW(im)
    for row_index, col_index in np.ndindex(imL.shape[0:2]):
        index = np.floor(imL[row_index, col_index, 0] * N)
        hist[index] += 1
    return hist / (imL.shape[0] * imL.shape[1])


def printHisto(im, N, scale):
    hist = histogram(im, N)
    for index in np.ndindex(hist.shape):
        print('X' * np.round(hist[index] * scale))

import numpy as np
from scipy import ndimage
import scipy
from scipy import signal
from numpy import exp

def check_module():
    return 'Cristina Lozano'

def boxBlur(im, k):
    ''' Return a  blured image filtered by box filter'''
    im_out = np.zeros((im.shape[0], im.shape[1], 3))
    step = (k-1)/2
    for y, x in imIter(im):
        ymin = clipY(im , y-step)
        ymax = clipY(im, y+step)
        xmin = clipX(im, x-step)
        xmax = clipX(im, x+step)
        box = im[ymin:ymax, xmin:xmax, :]
        im_out[y,x] = np.sum(box, (0, 1))/float(box.shape[0]*box.shape[1])
    return im_out

def convolve(im, kernel):
    ''' Return an image filtered by kernel'''
    xOffset = int(kernel.shape[1])/2
    yOffset = int(kernel.shape[0])/2
    im_out = np.zeros((im.shape[0], im.shape[1], 3))
    for y, x in imIter(im_out):
        for yP, xP in imIter(kernel):
            im_out[y,x] += np.dot(pix(im, y+yP-yOffset, x+xP-xOffset, True), kernel[yP, xP])
    return im_out

def gradientMagnitude(im):
    ''' Return the sum of the absolute value of the gradient  
    The gradient is the filtered image by Sobel filter '''
    Sobel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    imGH = convolve(im, Sobel)
    imGV = convolve(im, Sobel.transpose())
    im_out = np.sqrt(np.square(imGH) + np.square(imGV))
    return im_out

def horiGaussKernel(sigma, truncate=3):
    '''Return an one d kernel '''
    kernSize = int(sigma*truncate)*2 + 1
    some_array = np.zeros(kernSize)
    for x in range(some_array.shape[0]):
        dist = x-int(sigma*truncate)
        some_array[x] = np.exp((-1.0*np.square(dist))/(2.0*np.square(sigma)))
    aNorm = np.sum(some_array)
    some_array = np.array([some_array/aNorm])
    return some_array

def gaussianBlur(im, sigma, truncate=3):
    horiKernel = horiGaussKernel(sigma, truncate)
    vertKernel = horiKernel.transpose()
    out = convolve(im, horiKernel)
    gaussian_blurred_image = convolve(out, vertKernel)
    return gaussian_blurred_image


def gauss2D(sigma=2, truncate=3):
    '''Return an 2-D array of gaussian kernel'''
    kernSize = int(sigma*truncate)*2 + 1
    gaussian_kernel = np.zeros((kernSize, kernSize))
    for y, x in imIter(gaussian_kernel):
        dist = np.sqrt(np.square(x-int(sigma*truncate)) + np.square(y-int(sigma*truncate)))
        gaussian_kernel[y,x] = np.exp((-1.0*np.square(dist))/(2.0*np.square(sigma)))
    aNorm = np.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel/aNorm
    return gaussian_kernel 

def unsharpenMask(im, sigma, truncate, strength):
    blur = gaussianBlur(im, sigma, truncate)
    sharpened_image = im.copy()
    sharpened_image += (sharpened_image-blur)*strength
    return sharpened_image

def bilateral(im, sigmaRange, sigmaDomain):
    bilateral_filtered_image = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.float64)
    truncate = 2
    window = int(sigmaDomain * truncate)
    for y, x in imIter(bilateral_filtered_image):
        k = 0.0
        pixel = np.array([0,0,0], dtype=np.float64)
        for yP in xrange(y - window, y + window + 1):
            for xP in xrange(x - window, x + window + 1):
                imDist = np.sqrt(float((x-xP)**2 + (y-yP)**2))
                gIm = gaussEq(imDist, sigmaDomain, 2)
                colorDist = np.sqrt(float(np.sum(np.square(im[y,x]-pix(im, yP, xP, True)))))
                gColor = gaussEq(colorDist, sigmaRange, 2)
                k += gIm*gColor
                pixel += gIm*gColor*pix(im, yP, xP, True)
        bilateral_filtered_image[y,x] = pixel/k
    return bilateral_filtered_image


def bilaYUV(im, sigmaRange, sigmaY, sigmaUV):
    '''6.865 only: filter YUV differently'''
    bilateral_filtered_image = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.float64)
    truncate = 2
    windowY = int(sigmaY * truncate)
    windowUV = int(sigmaUV * truncate)
    imYUV = rgb2yuv(im)
    for y, x in imIter(bilateral_filtered_image):
        kY = 0.0
        pixelY = 0.0
        kUV = 0.0      
        pixelUV = np.array([0,0], dtype=np.float64)
        for yP in xrange(y - windowUV, y + windowUV + 1):
            for xP in xrange(x - windowUV, x + windowUV + 1):
                imDist = np.sqrt(float((x-xP)**2 + (y-yP)**2))
                gImUV = gaussEq(imDist, sigmaUV, 2)
                colorDist = np.sqrt(float(np.sum(np.square(imYUV[y,x]-pix(imYUV, yP, xP, True)))))
                gColor = gaussEq(colorDist, sigmaRange, 2)
                kUV += gImUV*gColor
                pixelUV += gImUV*gColor*pix(imYUV, yP, xP, True)[1:]
                if yP >= (y - windowY) and yP <= (y + windowY) and xP >= (x - windowY) and xP <= (x + windowY):
                    gImY = gaussEq(imDist, sigmaY, 2)
                    kY += gImY*gColor
                    pixelY += gImY*gColor*pix(imYUV, yP, xP, True)[0]
        bilateral_filtered_image[y,x,0] = pixelY/kY
        bilateral_filtered_image[y,x, 1:] = pixelUV/kUV
    return yuv2rgb(bilateral_filtered_image)

# Helpers

def gaussEq(radius, sigma=2, truncate=3):
    gaussian = np.exp((-1.0*radius**2)/(2.0*sigma**2))
    return gaussian

def pix(im, y, x, repeatEdge=False):
    if repeatEdge:
        if y < 0:
            y = 0
        elif y >= im.shape[0]:
            y = im.shape[0] - 1
        if x < 0:
            x = 0
        elif x >= im.shape[1]:
            x = im.shape[1] - 1 
    else:
        if y < 0 or x < 0 or y >= im.shape[0] or x >= im.shape[1]:
            return np.array((0,0,0))
    return im[y,x]
    
def clipX(im, x): 
    return min(im.shape[1]-1, max(x, 0))
    
def clipY(im, y): 
    return min(im.shape[0]-1, max(y, 0))
    
def imIter(im, debug=False, lim=1e6):
    for y in xrange(min(im.shape[0], lim)):
        if debug & (y%10==0): print 'y=', y
        for x in xrange(min(lim, im.shape[1])):
            yield y, x
            
def rgb2yuv(im):
    imyuv = im.copy()
    transM = np.array([0.299, 0.587, 0.114, -0.14713, -0.28886, 0.436, 0.615, -0.51499, -0.10001]).reshape((3,3))
    for row_index, col_index in np.ndindex(imyuv.shape[0:2]):
    	imyuv[row_index, col_index, :] = np.dot(transM, imyuv[row_index, col_index, :])
    return imyuv


def yuv2rgb(im):
    imrgb = im.copy()
    transM = np.array([[1, 0, 1.13983, 1, -0.39465, -0.58060, 1, 2.03211, 0]]).reshape((3,3))
    for row_index, col_index in np.ndindex(imrgb.shape[0:2]):
    	imrgb[row_index, col_index, :] = np.dot(transM, imrgb[row_index, col_index, :])
    return imrgb
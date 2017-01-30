# Assignment 5 for 6.815/865
# Submission: 
# Deadline:
# Your name: Cristina Lozano
# Reminder: 
# - Don't hand in data
# - Don't forget README.txt

import numpy as np
import bilagrid

def computeWeight(im, epsilonMini=0.002, epsilonMaxi=0.99):
    out = np.ones(im.shape)
    out[im<epsilonMini] = 0
    out[im>epsilonMaxi] = 0
    return out 

def computeFactor(im1, w1, im2, w2):
    wUsable = np.ones(im1.shape)
    wUsable[w1==0] = 0
    wUsable[w2==0] = 0
    im1 = im1.clip(0.000001)
    imRatio = im2/im1
    imRatio = imRatio*wUsable
    imRatio = imRatio[np.nonzero(imRatio)]
    factor = np.median(imRatio)
    return factor

def makeHDR(imageList, epsilonMini=0.002, epsilonMaxi=0.99):
    kj = 1.0
    imageList.reverse()
    weightSum = computeWeight(imageList[0], -1, epsilonMaxi)
    out = weightSum*imageList[0]
    for i in xrange(1, len(imageList)):
        im1 = imageList[i-1]
        im2 = imageList[i]
        if i == 1:
            w1 = computeWeight(im1, -1, epsilonMaxi)
        else:
            w1 = computeWeight(im1, epsilonMini, epsilonMaxi)
            
        if i == len(imageList)-1:
            w2 = computeWeight(im2, epsilonMini, 2)
        else:
            w2 = computeWeight(im2, epsilonMini, epsilonMaxi)
        k = computeFactor(im1, w1, im2, w2)
        ki = k*kj
        out += w2 / ki * im2
        kj = ki
        weightSum += w2
    weightSum = weightSum.clip(0.000001)
    out /= weightSum
    return out
    
def toneMap(im, targetBase=100, detailAmp=1, useBila=False):
    imL, imC = lumiChromi(im)
    minVal = numpymin(imL)
    imL = imL.clip(minVal)
    logImL = np.log10(imL)
    sigmaS = np.max(imC.shape)/50.0
    if useBila:
        base = bilagrid.bilateral_grid(imL, sigmaS, 0.4)
    else:
        base = ndimage.filters.gaussian_filter(imL, [sigmaS, sigmaS, 0])
    base = np.log10(base)
    detail = logImL-base
    dynamic_range = (np.max(base)-np.min(base))
    scaleFactor = targetBase/dynamic_range
    outLog = detailAmp*detail + scaleFactor*(base - np.max(base))
    output = 10**outLog
    return output


### HELPERS

def BW(im, weights=[0.3,0.6,0.1]):
    img = im.copy()
    greyVal =  np.dot(img[:,:],weights)
    img[:,:,0] = greyVal
    img[:,:,1] = greyVal
    img[:,:,2] = greyVal
    return img


def lumiChromi(im):
    imL = BW(im)
    imC = im.copy()/imL
    return (imL, imC)

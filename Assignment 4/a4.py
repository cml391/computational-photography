#assignment 4 starter code
#by Abe Davis
#
# Student Name:
# MIT Email:

import numpy as np

def denoiseSeq(imageList):
    '''Takes a list of images, returns a denoised image
    '''
    imD = np.zeros((imageList[0].shape[0], imageList[0].shape[1], 3), dtype=np.float64)
    for im in imageList:
        imD += im
    imD /= len(imageList)
    return imD

def logSNR(imageList, scale=1.0/20.0):
    '''takes a list of images and a scale. Returns an image showing log10(snr)*scale'''
    mean = denoiseSeq(imageList)
    variance = np.zeros((imageList[0].shape[0], imageList[0].shape[1], 3), dtype=np.float64)
    eVal = np.zeros((imageList[0].shape[0], imageList[0].shape[1], 3), dtype=np.float64)
    for im in imageList:
        variance += np.square(im - mean)
        eVal += np.square(im)
    variance /= (len(imageList) - 1)
    variance = variance.clip(0.00000001)
    eVal /= len(imageList)
    snr = eVal / variance
    snr = snr.clip(0.00000001)
    print 'PSNR ', np.max(snr)
    return np.log10(snr)*scale

def align(im1, im2, maxOffset=20):
    '''takes two images and a maxOffset. Returns the y, x offset that best aligns im2 to im1.'''
    minDiff = float('inf')
    y = 0
    x = 0
    ySize = im1.shape[0] - maxOffset
    xSize = im1.shape[1] - maxOffset
    for yOff in xrange(-maxOffset, maxOffset):
        imRollY = np.roll(im2, yOff, axis=0)
        for xOff in xrange(-maxOffset, maxOffset):
            imRollXY = np.roll(imRollY, xOff, axis=1)
            diff = np.sum(np.square(im1[maxOffset:ySize, maxOffset:xSize]-imRollXY[maxOffset:ySize, maxOffset:xSize]))
            if diff < minDiff:
                minDiff = diff
                y = yOff
                x = xOff
    return y, x

def alignAndDenoise(imageList, maxOffset=20):
    '''takes a list of images and a max offset. Aligns all of the images to the first image in the list, and averages to denoise. Returns the denoised image.'''
    alignImList = [imageList[0]]
    for im in imageList[1:]:
        y,x = align(imageList[0], im)
        alignIm = np.roll(im, y, 0)
        alignIm = np.roll(alignIm, x, 1)
        alignImList.append(alignIm)
    return denoiseSeq(alignImList)

def basicGreen(raw, offset=1):
    '''takes a raw image and an offset. Returns the interpolated green channel of your image using the basic technique.'''
    imInterp = raw.copy()
    for y in xrange(1, raw.shape[0]-1):
        for x in xrange(1, raw.shape[1]-1):
            if (y%2==offset and x%2==0) or (y%2!=offset and x%2==1):
                imInterp[y,x]=(imInterp[y-1,x]+imInterp[y+1,x]+imInterp[y,x-1]+imInterp[y,x+1])/4
    return imInterp
    

def basicRorB(raw, offsetY, offsetX):
    '''takes a raw image and an offset in x and y. Returns the interpolated red or blue channel of your image using the basic technique.'''
    imInterp = raw.copy()
    for y in xrange(1, raw.shape[0]-1):
        for x in xrange(1, raw.shape[1]-1):
            if (y%2!=offsetY and x%2==offsetX):
                imInterp[y,x]=(imInterp[y-1,x]+imInterp[y+1,x])/2
            elif (y%2==offsetY and x%2!=offsetX):
                imInterp[y,x]=(imInterp[y,x-1]+imInterp[y,x+1])/2
            elif (y%2!=offsetY and x%2!=offsetX):
                imInterp[y,x]=(imInterp[y-1,x-1]+imInterp[y-1,x+1]+imInterp[y+1,x-1]+imInterp[y+1,x+1])/4
    return imInterp

def basicDemosaic(raw, offsetGreen=0, offsetRedY=1, offsetRedX=1, offsetBlueY=0, offsetBlueX=0):
    '''takes a raw image and a bunch of offsets. Returns an rgb image computed with our basic techniche.'''
    imGreen = basicGreen(raw, offsetGreen)
    imRed = basicRorB(raw, offsetRedY, offsetRedX)
    imBlue = basicRorB(raw, offsetBlueY, offsetBlueX)
    rout = np.zeros((imGreen.shape[0], imGreen.shape[1], 3))
    for y,x in imIter(rout):
        rout[y,x,0] = imRed[y,x]
        rout[y,x,1] = imGreen[y,x]
        rout[y,x,2] = imBlue[y,x]
    return rout

def edgeBasedGreenDemosaic(raw, offsetGreen=0, offsetRedY=1, offsetRedX=1, offsetBlueY=0, offsetBlueX=0):
    '''same as basicDemosaic except it uses the edge based technique to produce the green channel.'''
    imGreen = edgeBasedGreen(raw, offsetGreen)
    imRed = basicRorB(raw, offsetRedY, offsetRedX)
    imBlue = basicRorB(raw, offsetBlueY, offsetBlueX)
    rout = np.zeros((imGreen.shape[0], imGreen.shape[1], 3))
    for y,x in imIter(rout):
        rout[y,x,0] = imRed[y,x]
        rout[y,x,1] = imGreen[y,x]
        rout[y,x,2] = imBlue[y,x]
    return rout

def edgeBasedGreen(raw, offset=1):
    '''same as basicGreen, but uses the edge based technique.'''
    imInterp = raw.copy()
    for y in xrange(1, raw.shape[0]-1):
        for x in xrange(1, raw.shape[1]-1):
            if (y%2==offset and x%2==0) or (y%2!=offset and x%2==1):
                yDiff=(imInterp[y-1,x]-imInterp[y+1,x])**2
                xDiff=(imInterp[y,x-1]-imInterp[y,x+1])**2
                if (yDiff < xDiff):
                    imInterp[y,x] = (imInterp[y-1,x]+imInterp[y+1,x])/2
                else:
                    imInterp[y,x] = (imInterp[y,x-1]+imInterp[y,x+1])/2
    return imInterp
    

def greenBasedRorB(raw, green, offsetY, offsetX):
    '''Same as basicRorB but also takes an interpolated green channel and uses this channel to implement the green based technique.'''
    rorB = basicRorB(raw-green, offsetY, offsetX)
    return rorB + green 

def improvedDemosaic(raw, offsetGreen=0, offsetRedY=1, offsetRedX=1, offsetBlueY=0, offsetBlueX=0):
    '''Same as basicDemosaic but uses edgeBasedGreen and greenBasedRorB.'''
    imGreen = edgeBasedGreen(raw, offsetGreen)
    imRed = greenBasedRorB(raw, imGreen, offsetRedY, offsetRedX)
    imBlue = greenBasedRorB(raw, imGreen, offsetBlueY, offsetBlueX)
    rout = np.zeros((imGreen.shape[0], imGreen.shape[1], 3))
    for y,x in imIter(rout):
        rout[y,x,0] = imRed[y,x]
        rout[y,x,1] = imGreen[y,x]
        rout[y,x,2] = imBlue[y,x]
    return rout

def split(raw):
    '''splits one of Sergei's images into a 3-channel image with height that is floor(height_of_raw/3.0). Returns the 3-channel image.'''
    cropHeight = raw.shape[0]/3
    out = np.zeros((cropHeight, raw.shape[1], 3))
    out[:,:,2] = raw[0:cropHeight, :]
    out[:,:,1] = raw[cropHeight:cropHeight*2, :]
    out[:,:,0] = raw[cropHeight*2:cropHeight*3, :]
    return out

def sergeiRGB(raw, alignTo=1):
    '''Splits the raw image, then aligns two of the channels to the third. Returns the aligned color image.'''
    cropHeight = raw.shape[0]/3
    out = np.zeros((cropHeight, raw.shape[1], 3))
    im1 = raw[0:cropHeight, :]
    im2 = raw[cropHeight:cropHeight*2, :]
    im3 = raw[cropHeight*2:cropHeight*3, :]
    out[:,:,0] = im3
    y,x = align(im3, im1)
    alignIm = np.roll(im1, y, 0)
    alignIm = np.roll(alignIm, x, 1)
    out[:,:,2] = alignIm
    y,x = align(im3, im2)
    alignIm = np.roll(im2, y, 0)
    alignIm = np.roll(alignIm, x, 1)
    out[:,:,1] = alignIm
    return out
    
#helpers
    
def imIter(im, debug=False, lim=1e6):
    for y in xrange(min(im.shape[0], lim)):
        if debug & (y%10==0): print 'y=', y
        for x in xrange(min(lim, im.shape[1])):
            yield y, x
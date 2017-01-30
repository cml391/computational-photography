#a2.py
import numpy as np
import math
import imageIO as io


#this file should only contain function definitions.
#It should not call the functions or perform any test.
#Do this in a separate file.

def check_my_module():
    ''' Fill your signature here. When upload your code, check if the signature is correct'''
    my_signature='Cristina Lozano'
    return my_signature 


def imIter(im, debug=False, lim=1e6):
    for y in xrange(min(im.shape[0], lim)):
        if debug & (y%10==0): print 'y=', y
        for x in xrange(min(lim, im.shape[1])):
            yield y, x


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


def scaleNN(im, k):
    '''Takes an image and a scale factor. Returns an image scaled using nearest neighbor interpolation.
    '''
    imS = np.zeros((round(im.shape[0]*k), round(im.shape[1]*k), 3))
    for y, x in imIter(imS):
        yInd = round(y/k)
        xInd = round(x/k)
        imS[y, x] = pix(im, yInd, xInd, True)
    return imS
    

def interpolateLin(im, y, x, repeatEdge=0):
    '''takes an image, y and x coordinates, and a bool
        returns the interpolated pixel value using bilinear interpolation
    '''
    yIndF = np.floor(y)
    yIndC = yIndF + 1
    yDiff = y - yIndF
    xIndF = np.floor(x)
    xIndC = xIndF + 1
    xDiff = x - xIndF
    topPix = (1-yDiff)*pix(im, yIndF, xIndF, repeatEdge) + (yDiff)*pix(im, yIndC, xIndF, repeatEdge)
    bottomPix = (1-yDiff)*pix(im, yIndF, xIndC, repeatEdge) + (yDiff)*pix(im, yIndC, xIndC, repeatEdge)
    pixel = (1-xDiff)*topPix + (xDiff)*bottomPix
    return pixel
    
        
        
def scaleLin(im, k):
    '''Takes an image and a scale factor. Returns an image scaled using bilinear interpolation.
    '''
    imS = np.zeros((round(im.shape[0]*k), round(im.shape[1]*k), 3))
    for y, x in imIter(imS):
        yInd = y/float(k)
        xInd = x/float(k)
        imS[y, x] = interpolateLin(im, yInd, xInd, True)
    return imS


def rotate(im, theta):
    '''takes an image and an angle in radians as input
        returns an image of the same size and rotated by theta
    '''
    imS = np.zeros((round(im.shape[0]), round(im.shape[1]), 3))
    yCenter = im.shape[0]/2
    xCenter = im.shape[1]/2
    for y, x in imIter(imS):
        cosTheta = math.cos(theta);
        sinTheta = math.sin(theta);
        xInd = (cosTheta * (x - xCenter) - sinTheta * (y - yCenter) + xCenter)
        yInd = (sinTheta * (x - xCenter) + cosTheta * (y - yCenter) + yCenter)
        imS[y, x] = interpolateLin(im, yInd, xInd)
    return imS
    

class segment:
    def __init__(self, x1, y1, x2, y2):
        #notice that the ui gives you x,y and we are storing as y,x
        self.P=np.array([y1, x1], dtype=np.float64)
        self.Q=np.array([y2, x2], dtype=np.float64)
        #You can precompute more variables here
        #...
        self.QP = (self.Q-self.P)
        self.magQP = self.QP[0]**2 + self.QP[1]**2
        self.perpQP = np.array([self.QP[1], -self.QP[0]], dtype=np.float64)
    

    def uv(self, X):
        '''Take the (y,x) coord given by X and return u, v values
        '''
        u = np.dot((X - self.P), self.QP) / self.magQP
        v = np.dot((X - self.P), self.perpQP) / math.sqrt(self.magQP)
        return u, v

    def dist (self, X):
        '''returns distance from point X to the segment (pill shape dist)
        '''
        if (self.magQP == 0):
            diffXP = X - self.P
            return math.sqrt(diffXP[1]**2 + diffXP[0]**2 )
        t = np.dot((X-self.P), self.QP) / self.magQP;
        if (t < 0): 
            diffXP = X - self.P
            return math.sqrt(diffXP[1]**2 + diffXP[0]**2 )
        elif (t > 1): 
            diffXQ = X - self.Q
            return math.sqrt(diffXQ[1]**2 + diffXQ[0]**2 )
        projection = self.P + t * self.QP
        diffXProj = X - projection
        return math.sqrt(diffXProj[1]**2 + diffXProj[0]**2 )
          
    def uvtox(self,u,v):
        '''take the u,v values and return the corresponding point (that is, the np.array([y, x]))
        '''
        return self.P + np.dot(u, self.QP) + np.dot(v, self.perpQP) / math.sqrt(self.magQP)
        

def warpBy1(im, segmentBefore, segmentAfter):
    '''Takes an image, one before segment, and one after segment. 
        Returns an image that has been warped according to the two segments.
    '''
    imD = np.zeros((round(im.shape[0]), round(im.shape[1]), 3))
    for y, x in imIter(imD):
        u,v = segmentAfter.uv(np.array([y,x]))
        xPrime = segmentBefore.uvtox(u,v)
        imD[y,x,:] = interpolateLin(im, xPrime[0], xPrime[1], True)
    return imD

def weight(s, X, a=10, b=1, p=1):
    '''Returns the weight of segment s on point X
    '''
    return (math.sqrt(s.magQP)**p / (a + s.dist(X)))**b

def warp(im, segmentsBefore, segmentsAfter, a=10, b=1, p=1):
    '''Takes an image, a list of before segments, a list of after segments, and the parameters a,b,p (see Beier)
    '''
    imD = np.zeros((round(im.shape[0]), round(im.shape[1]), 3))
    for y, x in imIter(imD):
        pixelX = np.array([y,x], dtype=np.float64)
        dsum = np.array([0,0], dtype=np.float64)
        weightsum = 0
        for i in range(len(segmentsAfter)):
            segAfter = segmentsAfter[i]
            segBefore = segmentsBefore[i]
            u,v = segAfter.uv(pixelX)
            xPrimeI = segBefore.uvtox(u,v)
            wght = weight(segAfter, pixelX, a, b, p)
            dsum += (xPrimeI - pixelX) * wght
            weightsum += wght
        xPrime = pixelX + (dsum / weightsum)
        imD[y,x,:] = interpolateLin(im, xPrime[0], xPrime[1], True)
    return imD

def morph(im1, im2, segmentsBefore, segmentsAfter, N=1, a=10, b=1, p=1):
    '''Takes two images, a list of before segments, a list of after segments, the number of morph images to create, and parameters a,b,p.
        Returns a list of images morphing between im1 and im2.
    '''
    sequence=list()
    sequence.append(im1.copy())
    for i in range(N):
        t = (i+1)/float(N+1)
        segments = list()
        for j in range(len(segmentsBefore)):
            segP = (1-t)*segmentsBefore[j].P + t*segmentsAfter[j].P
            segQ = (1-t)*segmentsBefore[j].Q + t*segmentsAfter[j].Q
            print segP
            segments.append(segment(segP[1], segP[0], segQ[1], segQ[0]))
        imW1 = warp(im1, segmentsBefore, segments, a, b, p)
        #io.imwrite(imW1, str(i) +'w1.png')
        imW2 = warp(im2, segmentsAfter, segments, a, b, p)
        #io.imwrite(imW2, str(i) +'w2.png')
        imM = (1-t)*imW1 + t*imW2
        sequence.append(imM)
    sequence.append(im2.copy())
    return sequence
    


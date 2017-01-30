#assignment 6 starter code
#by Abe Davis
#
# Student Name: cristina lozano
# MIT Email: clozano@mit.edu

import numpy as np
from scipy import linalg



def interpolateLin(im, y, x, repeatEdge=0):
    '''same as from previous assignment'''
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

def applyHomography(source, out, H, bilinear=False):
    '''takes the image source, warps it by the homography H, and adds it to the composite out. If bilinear=True use bilinear interpolation, otherwise use NN. Keep in mind that we are iterating through the output image, and the transformation from output pixels to source pixels is the inverse of the one from source pixels to the output. Does not return anything.'''
    hInv = np.linalg.inv(H)
    for y, x in imIter(out):
        homCoors = np.dot(hInv,np.array([y,x,1]))
        eucCoors = np.array([homCoors[0]/homCoors[2], homCoors[1]/homCoors[2]])
        if bilinear and pixInImage(source, np.round(eucCoors[0]), np.round(eucCoors[1])):
            print interpolateLin(source, eucCoors[0], eucCoors[1], True)
            out[y,x] = interpolateLin(source, eucCoors[0], eucCoors[1], True)
        else:
            eucCoors = np.round(eucCoors)
            if pixInImage(source, eucCoors[0], eucCoors[1]):
                out[y,x] = source[eucCoors[0], eucCoors[1]]

def addConstraint(systm, i, constr):
    '''Adds the constraint constr to the system of equations systm. constr is simply listOfPairs[i] from the argument to computeHomography. This function should fill in 2 rows of systm. We want the solution to our system to give us the elements of a homography that maps constr[0] to constr[1]. Does not return anything'''
    systm[i*2] = np.array([constr[0][0], constr[0][1], 1, 0, 0, 0, -constr[0][0]*constr[1][0], -constr[0][1]*constr[1][0], -constr[1][0]])
    systm[i*2+1] = np.array([0, 0, 0, constr[0][0], constr[0][1], 1, -constr[0][0]*constr[1][1], -constr[0][1]*constr[1][1], -constr[1][1]])
    
def computeHomography(listOfPairs):
    '''Computes and returns the homography that warps points listOfPairs[-][0] to listOfPairs[-][1]'''
    system = np.zeros((9,9), dtype=np.float64)
    system[8,8] = 1
    for i in range(4):
        addConstraint(system, i, listOfPairs[i])
    invSys = np.linalg.inv(system)
    B = np.zeros(9)
    B[8] = 1
    out = np.dot(invSys, B)
    return np.reshape(out, (3,3))

def computeTransformedBBox(imShape, H):
    '''computes and returns [[ymin, xmin],[ymax,xmax]] for the transformed version of the rectangle described in imShape. Keep in mind that when you usually compute H you want the homography that maps output pixels into source pixels, whereas here we want to transform the corners of our source image into our output coordinate system.'''
    yCoors = []
    xCoors = []
    homCoors = np.dot(H,np.array([0,0,1]))
    eucCoors = np.round(np.array([homCoors[0]/homCoors[2], homCoors[1]/homCoors[2]]))
    yCoors.append(eucCoors[0])
    xCoors.append(eucCoors[1])
    homCoors = np.dot(H,np.array([imShape[0],imShape[1],1]))
    eucCoors = np.round(np.array([homCoors[0]/homCoors[2], homCoors[1]/homCoors[2]]))
    yCoors.append(eucCoors[0])
    xCoors.append(eucCoors[1])
    homCoors = np.dot(H,np.array([imShape[0],0,1]))
    eucCoors = np.round(np.array([homCoors[0]/homCoors[2], homCoors[1]/homCoors[2]]))
    yCoors.append(eucCoors[0])
    xCoors.append(eucCoors[1])
    homCoors = np.dot(H,np.array([0,imShape[1],1]))
    eucCoors = np.round(np.array([homCoors[0]/homCoors[2], homCoors[1]/homCoors[2]]))
    yCoors.append(eucCoors[0])
    xCoors.append(eucCoors[1])
    yMax = max(yCoors)
    yMin = min(yCoors)
    xMax = max(xCoors)
    xMin = min(xCoors)
    return [[yMin, xMin],[yMax, xMax]]
    
def bboxUnion(B1, B2):
    '''No, this is not a professional union for beat boxers. Though that would be awesome. Rather, you should take two bounding boxes of the form [[ymin, xmin,],[ymax, xmax]] and compute their union. Return a new bounding box of the same form. Beat boxing optional...'''
    ymin = min(B1[0][0], B2[0][0])
    xmin = min(B1[0][1], B2[0][1])
    ymax = max(B1[1][0], B2[1][0])
    xmax = max(B1[1][1], B2[1][1])
    return [[ymin, xmin], [ymax, xmax]]

def translate(bbox):
    '''Takes a bounding box, returns a translation matrix that translates the top left corner of that bounding box to the origin. This is a very short function.'''
    return np.array([[1,0,-bbox[0][0]], [0,1,-bbox[0][1]], [0,0,1]])

def stitch(im1, im2, listOfPairs):
    '''Stitch im1 and im2 into a panorama. The resulting panorama should be in the coordinate system of im2, though possibly extended to a larger image. That is, im2 should never appear distorted in the resulting panorama, only possibly translated. Returns the stitched output (which may be larger than either input image).'''
    H = computeHomography(listOfPairs)
    BB1 = computeTransformedBBox(im1.shape, H)
    BB2 = [[0,0],[im2.shape[0], im2.shape[1]]]
    unionB = bboxUnion(BB1, BB2)
    transM = translate(unionB)
    out = np.zeros((unionB[1][0]-unionB[0][0], unionB[1][1]-unionB[0][1], 3))
    applyHomography(im2, out, transM)
    applyHomography(im1, out, np.dot(transM,H), True)
    return out

#######6.865 Only###############

def applyHomographyFast(source, out, H, bilinear=False):
    '''takes the image source, warps it by the homography H, and adds it to the composite out. This version should only iterate over the pixels inside the bounding box of source's image in out.'''
    hInv = np.linalg.inv(H)
    bbox = computeTransformedBBox(source.shape, H)
    for y in xrange(int(bbox[0][0]), int(bbox[1][0])):
        for x in xrange(int(bbox[0][1]), int(bbox[1][1])):
            homCoors = np.dot(hInv,np.array([y,x,1]))
            eucCoors = np.array([homCoors[0]/homCoors[2], homCoors[1]/homCoors[2]])
            if bilinear and pixInImage(source, np.round(eucCoors[0]), np.round(eucCoors[1])):
                out[y,x] = interpolateLin(source, eucCoors[0], eucCoors[1], True)
            else:
                eucCoors = np.round(eucCoors)
                if pixInImage(source, eucCoors[0], eucCoors[1]):
                    out[y,x] = source[eucCoors[0], eucCoors[1]]

def computeNHomographies(listOfListOfPairs, refIndex):
    '''This function takes a list of N-1 listOfPairs and an index. It returns a list of N homographies corresponding to your N images. The input N-1 listOfPairs describes all of the correspondences between images I(i) and I(i+1). The index tells you which of the images should be used as a reference. The homography returned for the reference image should be the identity.'''
    Hlist = []
    for listOfPairs in listOfListOfPairs:
        H = computeHomography(listOfPairs)
        Hlist.append(H)
    Hlist.insert(refIndex, np.array([[1,0,0],[0,1,0],[0,0,1]]))
    for i in reversed(xrange(0, refIndex)):
        Hlist[i]=np.dot(Hlist[i+1],Hlist[i])
    for i in xrange((refIndex+1), len(Hlist)):
        Hlist[i]=np.dot(Hlist[i-1],np.linalg.inv(Hlist[i]))
    return Hlist

def compositeNImages(listOfImages, listOfH):
    '''Computes the composite image. listOfH is of the form returned by computeNHomographies. Hint: You will need to deal with bounding boxes and translations again in this function.'''
    BB1 = computeTransformedBBox(listOfImages[0].shape, listOfH[0])
    BB2 = computeTransformedBBox(listOfImages[1].shape, listOfH[1])
    unionB = bboxUnion(BB1, BB2)
    for i in xrange(2, len(listOfH)):
        BBi = computeTransformedBBox(listOfImages[i].shape, listOfH[i])
        unionB = bboxUnion(unionB, BBi)
    transM = translate(unionB)
    out = np.zeros((unionB[1][0]-unionB[0][0], unionB[1][1]-unionB[0][1], 3))
    for i in xrange(len(listOfH)):
        applyHomographyFast(listOfImages[i], out, np.dot(transM,listOfH[i]), True)
    return out
    
def stitchN(listOfImages, listOfListOfPairs, refIndex):
    '''Takes a list of N images, a list of N-1 listOfPairs, and the index of a reference image. The listOfListOfPairs contains correspondences between each image Ii and image I(i+1). The function should return a completed panorama'''
    listOfH = computeNHomographies(listOfListOfPairs, refIndex)
    print listOfH
    pano = compositeNImages(listOfImages, listOfH)
    return pano
    
###### HELPERS ######

def pixInImage(im, y, x):
    if y < 0 or x < 0 or y >= im.shape[0] or x >= im.shape[1]:
        return False
    else:
        return True
        
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
    
def weight(s, X, a=10, b=1, p=1):
    '''Returns the weight of segment s on point X
    '''
    return (math.sqrt(s.magQP)**p / (a + s.dist(X)))**b
    
def imIter(im, debug=False, lim=1e6):
    for y in xrange(min(im.shape[0], lim)):
        if debug & (y%10==0): print 'y=', y
        for x in xrange(min(lim, im.shape[1])):
            yield y, x
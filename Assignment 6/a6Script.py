#script for running/testing assignment 6
#Starter code by Abe Davis
#
#
# Student Name: Cristina Lozano
# MIT Email: clozano@mit.edu

import a6
import numpy as np
import glob
import imageIO as io
from scipy import linalg
import time

def getPNGsInDir(path):
    fnames = glob.glob(path+"*.png")
    pngs = list()
    for f in fnames:
        #print f
        imi = io.getImage(f)
        pngs.append(imi)
    return pngs

def getRawPNGsInDir(path):
    fnames = glob.glob(path+"*.png")
    pngs = list()
    pngnames = list()
    print path
    for f in fnames:
        print f
        imi = io.imreadGrey(f)
        pngs.append(imi)
        pngnames.append(f)
    return pngs, pngnames

def testApplyHomographyPoster():
    signH = np.array([[1.12265192e+00, 1.44940136e-01, 1.70000000e+02], [8.65164180e-03, 1.19897030e+00, 9.50000000e+01],[  2.55704864e-04, 8.06420365e-04, 1.00000000e+00]])
    green = io.getImage("green.png")
    poster = io.getImage("poster.png")
    a6.applyHomography(poster, green, signH, True)
    io.imwrite(green, "HWDueAt9pm_applyHomography.png")


def testComputeAndApplyHomographyPoster():
    green = io.getImage("green.png")
    poster = io.getImage("poster.png")

    h, w = poster.shape[0]-1, poster.shape[1]-1
    pointListPoster=[np.array([0, 0, 1]), np.array([0, w, 1]), np.array([h, w, 1]), np.array([h, 0, 1])]
    pointListT=[np.array([170, 95, 1]), np.array([171, 238, 1]), np.array([233, 235, 1]), np.array([239, 94, 1])]

    listOfPairs=zip(pointListPoster, pointListT)
    
    H = a6.computeHomography(listOfPairs)
    #print H
    a6.applyHomography(poster, green, H, True)
    
    io.imwrite(green, "HWDueAt9pm_computeHomography.png")

def testFUN():
    sign = io.getImage("sign.png")
    funsign = io.getImage("fun-sign-no-alpha.png")
    
    h, w = funsign.shape[0]-1, funsign.shape[1]-1
    pointListFunSign=[np.array([0, 0, 1]), np.array([0, w, 1]), np.array([h, w, 1]), np.array([h, 0, 1])]
    pointListSign=[np.array([308, 274, 1]), np.array([271, 456, 1]), np.array([436, 304, 1]), np.array([397, 482, 1])]
    listOfPairs=zip(pointListFunSign, pointListSign)
    
    H = a6.computeHomography(listOfPairs)
    #print H
    print sign
    a6.applyHomography(funsign, sign, H, True)
    
    io.imwrite(sign, "Fun.png")
    
########

def testComputeAndApplyHomographyStata():
    im1=io.imread('stata/stata-1.png')
    im2=io.imread('stata/stata-2.png')
    pointList1=[np.array([209, 218, 1]), np.array([425, 300, 1]), np.array([209, 337, 1]), np.array([396, 336, 1])]
    pointList2=[np.array([232, 4, 1]), np.array([465, 62, 1]), np.array([247, 125, 1]), np.array([433, 102, 1])]
    listOfPairsS=zip(pointList1, pointList2)
    HS=a6.computeHomography(listOfPairsS)
    #multiply by 0.2 to better show the transition
    out=im2*0.5
    
    a6.applyHomography(im1, out, HS, True)
    io.imwrite(out, "stata_computeAndApplyHomography.png")

def testStitchStata():
    im1=io.imread('stata/stata-1.png')
    im2=io.imread('stata/stata-2.png')
    pointList1=[np.array([209, 218, 1]), np.array([425, 300, 1]), np.array([209, 337, 1]), np.array([396, 336, 1])]
    pointList2=[np.array([232, 4, 1]), np.array([465, 62, 1]), np.array([247, 125, 1]), np.array([433, 102, 1])]
    listOfPairs=zip(pointList1, pointList2)
    out = a6.stitch(im1, im2, listOfPairs)
    io.imwrite(out, "stata_stitch.png")

def testStitchScience():
    im1=io.imread('science/science-1.png')
    im2=io.imread('science/science-2.png')
    pointList1=[np.array([307, 15, 1], dtype=np.float64), np.array([309, 106, 1], dtype=np.float64), np.array([191, 102, 1], dtype=np.float64), np.array([189, 47, 1], dtype=np.float64)]
    pointList2=[np.array([299, 214, 1], dtype=np.float64), np.array([299, 304, 1], dtype=np.float64), np.array([182, 292, 1], dtype=np.float64), np.array([183, 236, 1], dtype=np.float64)]
    listOfPairs=zip(pointList1, pointList2)
    out = a6.stitch(im1, im2, listOfPairs)
    io.imwrite(out, "science_stitch.png")

def testStitchNStata():
    im1=io.imread('stata/stata-1.png')
    im2=io.imread('stata/stata-2.png')
    pointList1=[np.array([209, 218, 1]), np.array([425, 300, 1]), np.array([209, 337, 1]), np.array([396, 336, 1])]
    pointList2=[np.array([232, 4, 1]), np.array([465, 62, 1]), np.array([247, 125, 1]), np.array([433, 102, 1])]
    listOfPairs=zip(pointList1, pointList2)
    out = a6.stitchN([im1, im2], [listOfPairs],1)
    io.imwrite(out, "stata_N_stitch.png")
    
def testStitchMonu():
    im1=io.imread('monu/monu-1.png')
    im2=io.imread('monu/monu-2.png')
    pointList1=[np.array([121, 211, 1], dtype=np.float64), np.array([254, 252, 1], dtype=np.float64), np.array([147, 431, 1], dtype=np.float64), np.array([192, 316, 1], dtype=np.float64)]
    pointList2=[np.array([108, 5, 1], dtype=np.float64), np.array([260, 58, 1], dtype=np.float64), np.array([147, 238, 1], dtype=np.float64), np.array([190, 131, 1], dtype=np.float64)]
    listOfPairs=zip(pointList1, pointList2)
    out = a6.stitch(im1, im2, listOfPairs)
    io.imwrite(out, "MyPano.png")

def testStitchNVancouver():
    im1=io.imread('vancouverPan/vancouver0.png')
    im2=io.imread('vancouverPan/vancouver1.png')
    im3=io.imread('vancouverPan/vancouver2.png')
    im4=io.imread('vancouverPan/vancouver3.png')
    im5=io.imread('vancouverPan/vancouver4.png')
    
    pointList1=[np.array([117, 146, 1], dtype=np.float64), np.array([141, 70, 1], dtype=np.float64), np.array([187, 16, 1], dtype=np.float64), np.array([290, 123, 1], dtype=np.float64)]
    pointList2=[np.array([95, 295, 1], dtype=np.float64), np.array([128, 220, 1], dtype=np.float64), np.array([180, 171, 1], dtype=np.float64), np.array([281, 276, 1], dtype=np.float64)]
    listOfPairs=zip(pointList1, pointList2)
    listOfListOfPairs=[listOfPairs]
    
    pointList1=[np.array([235, 37, 1], dtype=np.float64), np.array([157, 83, 1], dtype=np.float64), np.array([242, 200, 1], dtype=np.float64), np.array([416, 67, 1], dtype=np.float64)]
    pointList2=[np.array([238, 161, 1], dtype=np.float64), np.array([162, 201, 1], dtype=np.float64), np.array([242, 323, 1], dtype=np.float64), np.array([415, 181, 1], dtype=np.float64)]
    listOfPairs=zip(pointList1, pointList2)
    listOfListOfPairs.append(listOfPairs)

    pointList1=[np.array([165, 40, 1], dtype=np.float64), np.array([417, 176, 1], dtype=np.float64), np.array([251, 97, 1], dtype=np.float64), np.array([294, 216, 1], dtype=np.float64)]
    pointList2=[np.array([157, 153, 1], dtype=np.float64), np.array([411, 269, 1], dtype=np.float64), np.array([241, 202, 1], dtype=np.float64), np.array([291, 326, 1], dtype=np.float64)]
    listOfPairs=zip(pointList1, pointList2)
    listOfListOfPairs.append(listOfPairs)
    
    pointList1=[np.array([154, 150, 1], dtype=np.float64), np.array([232, 68, 1], dtype=np.float64), np.array([165, 14, 1], dtype=np.float64), np.array([303, 27, 1], dtype=np.float64)]
    pointList2=[np.array([149, 304, 1], dtype=np.float64), np.array([238, 218, 1], dtype=np.float64), np.array([178, 172, 1], dtype=np.float64), np.array([303, 183, 1], dtype=np.float64)]
    listOfPairs=zip(pointList1, pointList2)
    listOfListOfPairs.append(listOfPairs)
    
    listOfImages = [im1, im2, im3, im4, im5]
    t=time.time()
    pano = a6.stitchN(listOfImages, listOfListOfPairs, 2)
    print time.time()-t, 'seconds'
    io.imwrite(pano, "pano_stitch.png")

#testApplyHomographyPoster()
#testComputeAndApplyHomographyPoster()
#testComputeAndApplyHomographyStata()
#testStitchStata()
#testStitchScience()
#testStitchNVancouver()
#testStitchMonu()
testFUN()

#***You can test on the first N images of a list by feeding im[:N] as the argument instead of im***


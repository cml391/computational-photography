#script for running/testing assignment 4
#Starter code by Abe Davis
#
#
# Student Name:
# MIT Email:

import a4
import numpy as np
import glob
import imageIO as io

io.baseInputPath = './'

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

#def getFirstN(imList, N):
#    '''Super simple function. I'm only including it as a reminder that you can test on a subset of the data. Python is slow...'''
#    return imList[:N]

def testAlign():
    print 'testAlign'
    align1 = np.random.rand(50,50,3)
    align2 = np.random.rand(50,50,3)
    align2[5:50, 3:50, :] = align1[0:45, 0:47, :]
    yalign, xalign = a4.align(align1, align2)
    yalign2, xalign2 = a4.align(align2, align1)
    print "alignment 1->2 is:"+'[{},{}]'.format(yalign, xalign)
    print "alignment 2->1 is:"+'[{},{}]'.format(yalign2, xalign2)



def testDenoise(imageList, outputname):
    print 'denoise'
    imdenoise = a4.denoiseSeq(imageList)
    io.imwrite(imdenoise, str(outputname+'_denoise_x%03d'%(len(imageList)))+".png")
    
def testSNR(imageList, outputname):
    print 'SNR'
    imSNR = a4.logSNR(imageList)
    io.imwrite(imSNR, str(outputname+'_logSNR_x%03d'%(len(imageList)))+".png")

def testAlignAndDenoise(imageList, outputname):
    print 'testAlignAndDenoise'
    imADN = a4.alignAndDenoise(imageList)
    io.imwrite(imADN, str(outputname+'_ADN_x%03d'%(len(imageList)))+".png")

def testBasicDemosaic(raw, outputname, gos =1, rosy=1,rosx=1,bosy=0,bosx=0):
    print 'testBasicDemosaic'
    rout = a4.basicDemosaic(raw, gos, rosy, rosx, bosy, bosx)
    io.imwrite(rout, outputname+'_basicDemosaic.png') 
    
def testBasicGreen(raw, outputname, gos =1):
    print 'testBasicGreen'
    rout = a4.basicGreen(raw, gos)
    io.imwriteGrey(rout, outputname+'_basicGreen.png')
    io.imwriteGrey(raw, outputname+'_basicGreenOrig.png')
    
def testBasicRedAndBlue(raw, outputname):
    print 'testBasicRedAndBlue'
    rout1 = a4.basicRorB(raw, 1, 1)
    rout2 = a4.basicRorB(raw, 0, 0)
    io.imwriteGrey(rout1, outputname+'_basicRed.png')
    io.imwriteGrey(rout2, outputname+'_basicBlue.png')

def testEdgeBasedGreenDemosaic(raw, outputname, gos =1, rosy=1,rosx=1,bosy=0,bosx=0):
    print 'testEdgeBasedGreenDemosaic'
    rout = a4.edgeBasedGreenDemosaic(raw, gos, rosy, rosx, bosy, bosx)
    io.imwrite(rout, outputname+'_edgeBasedGreenDemosaic.png') 

def testEdgeBasedGreen(raw, outputname, gos =1):
    print 'testEdgeBasedGreen'
    rout = a4.edgeBasedGreen(raw, gos)
    io.imwriteGrey(rout, outputname+'_edgeBasedGreen.png')

def testImprovedDemosaic(raw, outputname, gos =1, rosy=1,rosx=1,bosy=0,bosx=0):
    print 'testImprovedDemosaic'
    rout = a4.improvedDemosaic(raw, gos, rosy, rosx, bosy, bosx)
    io.imwrite(rout, outputname+'_improvedDemosaic.png') 

def testSergei():
    print 'testSergei'
    sergeis, sergeiNames = getRawPNGsInDir("data/Sergei/")
    scount = 0
    for f in sergeis:
        io.imwrite(a4.split(f), str('Split'+'%03d'%(scount))+'.png')
        io.imwrite(a4.sergeiRGB(f), str('Sergei'+'%03d'%(scount))+'.png')
        scount = scount +1



#Input data:
#
#Archive_2 = getPNGsInDir("Archive_2/")
#iso400 = getPNGsInDir("data/aligned-ISO400-16/")
#iso3200 = getPNGsInDir("data/aligned-ISO3200-16/")
#green = getPNGsInDir("data/green-9/")
#raw, rawnames = getRawPNGsInDir("data/raw/")
signsm = io.imreadGrey("data/raw/signs-small.png")

#***You can test on the first N images of a list by feeding im[:N] as the argument instead of im***

#example for testing denoise and logSNR
#testDenoise(iso400, "iso400")
#testDenoise(iso3200, "iso3200")

#testSNR(iso400, "iso400")
#testSNR(iso3200, "iso3200")

#test alignAndDenoise
#testAlignAndDenoise(green, 'green')
#testDenoise(green, 'green')

#testAlign()

testBasicGreen(signsm, 'signSmall', 0)
testBasicRedAndBlue(signsm, 'signSmall')
testBasicDemosaic(signsm, 'signSmall', 0)
testEdgeBasedGreen(signsm, 'signSmall', 0)
testEdgeBasedGreenDemosaic(signsm, 'signSmall', 0)
testImprovedDemosaic(signsm, 'signSmall', 0)

testSergei()

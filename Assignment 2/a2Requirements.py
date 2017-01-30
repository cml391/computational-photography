import a2
import imageIO
import numpy
import random
import unittest
import math

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.im = imageIO.imread()
        self.imPanda = imageIO.imread('panda2.png')
        self.imBear = imageIO.imread('bear.png')
        self.imFredo = imageIO.imread('fredo2.png')
        self.imWolf = imageIO.imread('werewolf.png')

    def test_0_imageLoad(self):
        self.assertEqual(self.im.shape, (85, 128, 3), "Size of input image is wrong. Have you modified in.png by accident?")
    '''
    def test_1_scaleNN(self):
        imS = a2.scaleNN(self.im, 2.3)
        imageIO.imwrite(imS, 'scaleNN.png')
        
    def test_2_scaleLin(self):
        imS = a2.scaleLin(self.im, 3)
        imageIO.imwrite(imS, 'scaleLin.png')
     
    def test_3_testRotate(self):
        imR = a2.rotate(self.imPanda, math.pi/4)
        imageIO.imwrite(imR, 'rotate.png')
      
    def test_4_dist(self):
        imD = numpy.zeros((100,100,3))
        seg = a2.segment(30, 0, 70, 0)
        for y, x in a2.imIter(imD, debug=True):
            color = seg.dist(numpy.array([y,x]))
            color = color/200.0
            imD[y,x,:] = color
        imageIO.imwrite(imD, 'dist.png')
    
    
    def test_5_warpBy1(self):
        imW = a2.warpBy1(self.imBear, a2.segment(0,0, 10,0), a2.segment(10, 10, 30, 15))
        imageIO.imwrite(imW, 'warp1Bear.png')  
    
    def test_6_warp(self):
        imW = a2.warp(self.imBear, [a2.segment(0,0, 10,0)], [a2.segment(10, 10, 30, 15)])
        imageIO.imwrite(imW, 'warpBear.png') 
    
    
    def test_7_morph(self):
        segmentsBefore=numpy.array([a2.segment(18, 49, 43, 29), a2.segment(153, 32, 170, 67), a2.segment(86, 129, 109, 127), a2.segment(147, 129, 165, 133), a2.segment(101, 201, 135, 194), a2.segment(118, 226, 135, 208), a2.segment(48, 209, 70, 242), a2.segment(143, 235, 158, 183)])
        segmentsAfter=numpy.array([a2.segment(20, 38, 38, 13), a2.segment(147, 17, 165, 53), a2.segment(85, 109, 105, 110), a2.segment(138, 98, 158, 107), a2.segment(100, 175, 148, 158), a2.segment(115, 198, 143, 184), a2.segment(81, 184, 106, 216), a2.segment(147, 195, 154, 147)])
        imSeq = a2.morph(self.imFredo, self.imWolf, segmentsBefore, segmentsAfter, 2) 
        for i in range(len(imSeq)):
            imageIO.imwrite(imSeq[i], str(i)+'.png')
    '''
    def test_8_morph(self):
        me = imageIO.imread('classMorph_25.png')
        stephanie = imageIO.imread('classMorph_26.png')
        segmentsBefore=numpy.array([a2.segment(90, 154, 99, 160), a2.segment(107, 161, 115, 153), a2.segment(55, 170, 84, 203), a2.segment(114, 206, 141, 173), a2.segment(40, 79, 78, 46), a2.segment(117, 42, 141, 70), a2.segment(66, 127, 89, 127), a2.segment(108, 127, 131, 121), a2.segment(59, 115, 85, 112), a2.segment(109, 114, 136, 110), a2.segment(36, 127, 47, 162), a2.segment(154, 116, 148, 151), a2.segment(80, 170, 99, 183), a2.segment(124, 169, 108, 184), a2.segment(88, 168, 115, 168), a2.segment(51, 181, 55, 213), a2.segment(135, 191, 135, 213)])
        segmentsAfter=numpy.array([a2.segment(86, 150, 99, 157), a2.segment(105, 157, 116, 150), a2.segment(58, 171, 90, 204), a2.segment(114, 202, 147, 173), a2.segment(47, 75, 78, 43), a2.segment(117, 38, 147, 64), a2.segment(67, 123, 88, 123), a2.segment(115, 121, 136, 118), a2.segment(61, 105, 93, 102), a2.segment(112, 103, 135, 100), a2.segment(42, 119, 50, 156), a2.segment(159, 112, 154, 151), a2.segment(85, 169, 103, 183), a2.segment(120, 168, 111, 181), a2.segment(93, 169, 113, 169), a2.segment(29, 162, 25, 203), a2.segment(175, 175, 184, 200)])
        imSeq = a2.morph(me, stephanie, segmentsBefore, segmentsAfter, 3) 
        for i in range(len(imSeq)):
            imageIO.imwrite(imSeq[i], 'classMorphHAIR_25_%02d.png' % i)
 
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
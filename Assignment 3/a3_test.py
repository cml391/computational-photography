from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import a3
import matplotlib.cm as cm
import imageIO as myImageIO 
import time

## No need to touch this class
#class myImageIO:
#  @staticmethod
#  def imread(path='in.png'):
#    from scipy import misc
#    return (misc.imread(path).astype(float)/255)**2.2
#
#  @staticmethod
#  def imreadg(path='in.png'):
#    from scipy import misc
#    return (misc.imread(path).astype(float)/255)

#  @staticmethod
#  def imwrite(im_in, path):
#    from scipy import misc
#    im_in[im_in<0]=0
#    im_in[im_in>1]=1
#    return misc.imsave(path, (255*(im_in**(1/2.2))).astype('uint8))

#  @staticmethod
#  def imwriteg(im_in, path):
#    from scipy import misc
#    im_in[im_in<0]=0
#    im_in[im_in>1]=1
#    return misc.imsave(path, (255*im_in).astype('uint8'))
  
#  @staticmethod
#  def thresh(im_in):
#    im_in[im_in<0]=0
#    im_in[im_in>1]=1
#    return im_in 

## Test case ## 
## Feel free to change the parameters or use the impulse as input

def test_box_blur():
  im=myImageIO.imread('images/pru.png')
  #im = impulse()
  out=a3.boxBlur(im, 7)
  myImageIO.imwrite(out, 'my_boxblur.png')

def test_convolve_gauss():

  im=myImageIO.imread('images/pru.png')
  gauss3=np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
  kernel=gauss3.astype(float)
  kernel=kernel/sum(sum(kernel))
  out=a3.convolve(im, kernel)
  myImageIO.imwrite(out, 'my_convolve_gaussblur.png')

def test_convolve_deriv():
  im=myImageIO.imread('images/pru.png')
  deriv=np.array([[-1, 1]])
  out=a3.convolve(im, deriv)
  myImageIO.imwrite(out, 'my_deriv.png')

def test_convolve_Sobel():
  im=myImageIO.imread('images/pru.png')
  Sobel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  out=a3.convolve(im, Sobel)
  myImageIO.imwrite(out, 'my_Sobel.png')

def test_grad():
  im=myImageIO.imread('images/pru.png')
  out=a3.gradientMagnitude(im)
  myImageIO.imwrite(out, 'my_gradient.png')

def test_horigauss():
  im=myImageIO.imread('images/pru.png')
  #im = impulse()
  kernel=a3.horiGaussKernel(2,3)
  out=a3.convolve(im, kernel)
  myImageIO.imwrite(out, 'my_horigauss.png')

def test_gaussianBlur():
  im=myImageIO.imread('images/pru.png')
  #im = impulse()
  t=time.time()
  out=a3.gaussianBlur(im, 2,3)
  print time.time()-t, 'seconds'
  myImageIO.imwrite(out, 'my_gaussBlur.png')


def test_gauss2D():
  im=myImageIO.imread('images/pru.png')
  #im = impulse()
  t=time.time()
  out=a3.convolve(im, a3.gauss2D())
  print time.time()-t, 'seconds'
  myImageIO.imwrite(out, 'my_gauss2DBlur.png')

def test_equal():
  im=myImageIO.imread('images/pru.png')
  out1=a3.convolve(im, a3.gauss2D())
  out2=a3.gaussianBlur(im,2, 3)
  res=abs(out1-out2);
  return (sum(res.flatten())<0.1)
  
def test_unsharpen():
  im=myImageIO.imread('images/zebra.png')
  out=a3.unsharpenMask(im, 1, 3, 1)
  myImageIO.imwrite(out, 'my_unsharpen.png')

def test_bilateral():
  im=myImageIO.imread('images/lens-3-med.png', 1.0)
  out=a3.bilateral(im, 0.1, 1.4)
  myImageIO.imwrite(out, 'my_bilateral.png', 1.0)
  

def test_bilaYUV():
  im=myImageIO.imread('images/lens-3-small.png', 1.0)
  out=a3.bilaYUV(im, 0.1, 1.4, 6)
  myImageIO.imwrite(out, 'my_bilaYUV.png', 1.0)
  
def impulse(h=100, w=100):
    out=np.zeros((h, w, 3))
    out[h/2, w/2]=1
    return out


#Uncomment the following function to test your code


#test_box_blur()
#test_convolve_gauss()
#test_convolve_deriv()
#test_convolve_Sobel()
#test_grad()
#test_horigauss()
test_gaussianBlur()
test_gauss2D()
#print test_equal()
#test_unsharpen()
#test_bilateral()
#test_bilaYUV()

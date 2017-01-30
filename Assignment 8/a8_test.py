from utils import imageIO as io
import a8
import numpy as np 

def test_dot():
  im1 = np.array([[[1,2,3],[2,2,2]], [[4,7,2], [3,2,3]], [[2,1,1.5], [3,4,5]]])
  im2 = np.array([[[3,24,1],[1,4,5]], [[3,3,3], [2,1,4]], [[2,4,8], [11,2,4]]])
  # should be 214
  print a8.dotIm(im1,im2)

def test_grad_descent():
  im=io.imread('data/pru.png')
  kernel=a8.gauss2D(1)
  im_blur=a8.convolve3(im, kernel)

  io.imwrite(im_blur, 'pru_blur.png')
  im_sharp=a8.deconvGradDescent(im_blur, kernel);
  io.imwrite(im_sharp, 'pru_sharp.png')

def test_conjugate_grad_descent():
  im=io.imread('data/pru.png')
  kernel=a8.gauss2D(1)
  im_blur=a8.convolve3(im, kernel)

  io.imwrite(im_blur, 'pru_blur.png')
  im_sharp=a8.deconvCG(im_blur, kernel);
  io.imwrite(im_sharp, 'pru_sharp_CG.png')

def test_real_psf():
  im=io.imread('data/pru.png')
  f=open('psf', 'r')
  psf=[map(float, line.split(',')) for line in f ]
  kernel=np.array(psf)
  im_blur=a8.convolve3(im, kernel)
  #kernel=kernel[::-1, ::-1]
  io.imwrite(im_blur, 'pru_blur_real.png')
  io.imwriteGrey(kernel/np.max(kernel), 'psf.png')
  im_sharp=a8.deconvCG(im_blur, kernel, 4);
  io.imwrite(im_sharp, 'pru_sharp_CG_real.png')



def test_conjugate_grad_descent_reg():
  im=io.imread('data/pru.png')
  kernel=a8.gauss2D(1)
  im_blur=a8.convolve3(im, kernel)
  noise=np.random.random(im_blur.shape)-0.5
  im_blur_noisy=im_blur+0.05*noise

  io.imwrite(im_blur_noisy, 'pru_blur_noise.png')
  im_sharp=a8.deconvCG_reg(im_blur_noisy, kernel);
  im_sharp_wo_reg=a8.deconvCG(im_blur_noisy, kernel);
  io.imwrite(im_sharp, 'pru_sharp_CG_reg.png')
  io.imwrite(im_sharp_wo_reg, 'pru_sharp_CG_wo_reg.png')


def test_naive_composite():
  fg=io.imread('data/bear.png')
  bg=io.imread('data/waterpool.png')
  mask=io.imread('data/mask.png')
  out=a8.naiveComposite(bg, fg, mask, 50, 1)
  io.imwrite(out, 'naive_composite.png')

def test_Poisson():

  y=50
  x=10
  useLog=True

  fg=io.imread('data/bear.png')
  bg=io.imread('data/waterpool.png')
  mask=io.imread('data/mask.png')


  h, w=fg.shape[0], fg.shape[1]
  mask[mask>0.5]=1.0
  mask[mask<0.6]=0.0
  bg2=(bg[y:y+h, x:x+w]).copy()
  out=bg.copy()
  if useLog:
      bg2[bg2==0]=1e-4
      fg[fg==0]=1e-4
      bg3=np.log(bg2)+3
      fg3=np.log(fg)+3
  else:
      bg3=bg2
      fg3=fg
  tmp=a8.Poisson(bg3, fg3, mask, 600)

  if useLog:
      out[y:y+h, x:x+w]=np.exp(tmp-3)
  else: out[y:y+h, x:x+w]=tmp

  io.imwrite(out, 'poisson.png')

def test_PoissonCG():

  y=50
  x=10
  useLog=True

  fg=io.imread('data/bear.png')
  bg=io.imread('data/waterpool.png')
  mask=io.imread('data/mask.png')


  h, w=fg.shape[0], fg.shape[1]
  mask[mask>0.5]=1.0
  mask[mask<0.6]=0.0
  bg2=(bg[y:y+h, x:x+w]).copy()
  out=bg.copy()
  if useLog:
      bg2[bg2==0]=1e-4
      fg[fg==0]=1e-4
      bg3=np.log(bg2)+3
      fg3=np.log(fg)+3
  else:
      bg3=bg2
      fg3=fg
  tmp=a8.PoissonCG(bg3, fg3, mask)

  if useLog:
      out[y:y+h, x:x+w]=np.exp(tmp-3)
  else: out[y:y+h, x:x+w]=tmp

  io.imwrite(out, 'poisson_CG.png')
  
def test_my_PoissonCG():

  y=0
  x=0
  useLog=True

  fg=io.imread('data/my_bears.png')
  bg=io.imread('data/field.png')
  mask=io.imread('data/my_mask.png')


  h, w=fg.shape[0], fg.shape[1]
  mask[mask>0.5]=1.0
  mask[mask<0.6]=0.0
  bg2=(bg[y:y+h, x:x+w]).copy()
  out=bg.copy()
  if useLog:
      bg2[bg2==0]=1e-4
      fg[fg==0]=1e-4
      bg3=np.log(bg2)+3
      fg3=np.log(fg)+3
  else:
      bg3=bg2
      fg3=fg
  tmp=a8.PoissonCG(bg3, fg3, mask, 600)

  if useLog:
      out[y:y+h, x:x+w]=np.exp(tmp-3)
  else: out[y:y+h, x:x+w]=tmp

  io.imwrite(out, 'my_poisson_CG.png')


#test_dot()
#test_grad_descent()
#test_conjugate_grad_descent()
#test_real_psf()
#test_conjugate_grad_descent_reg()
#test_naive_composite()
#test_Poisson()
test_my_PoissonCG()


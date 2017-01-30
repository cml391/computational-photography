import numpy as np

# === Deconvolution with gradient descent ===

def dotIm(im1, im2):
  return np.sum(im1*im2)

def applyKernel(im, kernel):
  ''' return Mx, where x is im '''
  return convolve3(im, kernel)

def applyConjugatedKernel(im, kernel):
  ''' return M^T x, where x is im '''
  M_T = np.flipud(np.fliplr(kernel))
  return applyKernel(im, M_T)

def computeResidual(kernel, x, y):
  ''' return y-Mx '''
  return y - applyKernel(x, kernel)

def computeStepSize(r, kernel):
  Mr = applyKernel(r, kernel)
  alpha = dotIm(r,r)/dotIm(r, applyConjugatedKernel(Mr, kernel))
  return alpha

def deconvGradDescent(im_blur, kernel, niter=10):
  ''' return deblurred image '''
  im = im_blur.copy()
  for i in xrange(niter):
    ri = applyConjugatedKernel(computeResidual(kernel, im, im_blur), kernel)
    im += ri*computeStepSize(ri, kernel)
  return im

# === Deconvolution with conjugate gradient ===

def computeGradientStepSize(r, d, kernel):
  Mr = applyKernel(d, kernel)
  alpha = dotIm(r,r)/dotIm(d, applyConjugatedKernel(Mr, kernel))
  return alpha

def computeConjugateDirectionStepSize(old_r, new_r):
  return dotIm(new_r, new_r)/dotIm(old_r, old_r)

def deconvCG(im_blur, kernel, niter=10):
  ''' return deblurred image '''
  im = im_blur.copy()
  ri = applyConjugatedKernel(computeResidual(kernel, im, im_blur), kernel)
  di = ri
  for i in xrange(niter):
    alpha = computeGradientStepSize(ri, di, kernel)
    im += di*alpha
    riplus1 = ri - alpha*applyConjugatedKernel(applyKernel(di, kernel), kernel)
    beta = computeConjugateDirectionStepSize(ri, riplus1)
    di = riplus1 + beta*di
    ri = riplus1
  return im

def laplacianKernel():
  ''' a 3-by-3 array '''
  laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
  return laplacian_kernel

def applyLaplacian(im):
  ''' return Lx (x is im)'''
  return applyKernel(im, laplacianKernel())

def applyAMatrix(im, kernel):
  ''' return Ax, where A = M^TM'''
  out = applyKernel(im, kernel)
  return applyConjugatedKernel(out, kernel)

def applyRegularizedOperator(im, kernel, lamb):
  ''' (A + lambda L )x'''
  return applyAMatrix(im, kernel) + lamb*applyLaplacian(im)

def computeGradientStepSize_reg(grad, p, kernel, lamb):
  alpha = dotIm(grad, grad)/dotIm(p, applyRegularizedOperator(p, kernel, lamb))
  return alpha

def deconvCG_reg(im_blur, kernel, lamb=0.05, niter=10):
  ''' return deblurred and regularized im '''
  im = im_blur.copy()
  ri = applyConjugatedKernel(im_blur, kernel) - applyRegularizedOperator(im, kernel, lamb)
  di = ri
  for i in xrange(niter):
    alpha = computeGradientStepSize_reg(ri, di, kernel, lamb)
    im += di*alpha
    riplus1 = ri - alpha*applyRegularizedOperator(di, kernel, lamb)
    beta = computeConjugateDirectionStepSize(ri, riplus1)
    di = riplus1 + beta*di
    ri = riplus1
  return im
    
def naiveComposite(bg, fg, mask, y, x):
  ''' naive composition'''
  inv_mask = np.ones(mask.shape) - mask
  out = bg.copy()
  out[y:y+fg.shape[0], x:x+fg.shape[1],:] = inv_mask*out[y:y+fg.shape[0], x:x+fg.shape[1],:] + mask*fg
  return out

def Poisson(bg, fg, mask, niter=200):
  ''' Poisson editing using gradient descent'''
  inv_mask = np.ones(mask.shape) - mask
  im = bg.copy()*inv_mask
  b = applyLaplacian(fg)
  for i in xrange(niter):
    ri = (b - applyLaplacian(im))*mask
    im += ri*computeStepSize(ri, laplacianKernel())
  return im 



def PoissonCG(bg, fg, mask, niter=200):
  ''' Poison editing using conjugate gradient '''
  inv_mask = np.ones(mask.shape) - mask
  im = bg.copy()*inv_mask
  b = applyLaplacian(fg)
  ri = (b - applyLaplacian(im))*mask
  di = ri
  for i in xrange(niter):
    alpha = dotIm(ri,ri)/dotIm(di, applyLaplacian(di))
    im += di*alpha
    riplus1 = (ri - alpha*applyLaplacian(di))*mask
    beta = computeConjugateDirectionStepSize(ri, riplus1)
    di = riplus1 + beta*di
    ri = riplus1
  return im
  

#==== Helpers. Use them as possible. ==== 

def convolve3(im, kernel):
  from scipy import ndimage
  center=(0,0)
  r=ndimage.filters.convolve(im[:,:,0], kernel, mode='reflect', origin=center) 
  g=ndimage.filters.convolve(im[:,:,1], kernel, mode='reflect', origin=center) 
  b=ndimage.filters.convolve(im[:,:,2], kernel, mode='reflect', origin=center) 
  return (np.dstack([r,g,b]))

def gauss2D(sigma=2, truncate=3):
  kernel=horiGaussKernel(sigma, truncate);
  kerker=np.dot(kernel.transpose(), kernel)
  return kerker/sum(kerker.flatten())

def horiGaussKernel(sigma, truncate=3):
  from scipy import signal
  sig=signal.gaussian(2*int(sigma*truncate)+1,sigma)
  return np.array([sig/sum(sig)])




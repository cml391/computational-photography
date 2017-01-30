import numpy as np
import scipy
from scipy import ndimage
import random as rnd
import a6
from utils import imageIO as io

class point():
  def __init__(self, x, y):
    self.x=x
    self.y=y

class feature():
  def __init__(self, pt, descriptor):
    self.pt=pt
    self.descriptor=descriptor

class correspondence():
  def __init__(self, pt1, pt2):
    self.pt1=pt1
    self.pt2=pt2

def computeTensor(im, sigmaG=1, factorSigma=4):
  '''im_out: 3-channel-2D array. The three channels are Ixx, Ixy, Iyy'''
  im_lum = BW(im)
  im_blur = ndimage.filters.gaussian_filter(im_lum, [sigmaG, sigmaG])
  im_gradX,im_gradY = gradientXY(im_blur)
  im_out = np.zeros((im.shape[0], im.shape[1], 3))
  im_out[:,:,0] = np.square(im_gradX)
  im_out[:,:,1] = im_gradX*im_gradY
  im_out[:,:,2] = np.square(im_gradY)
  im_blur_out = ndimage.filters.gaussian_filter(im_out, [sigmaG*factorSigma, sigmaG*factorSigma, 0])
  return im_blur_out


def cornerResponse(im, k=0.15, sigmaG=1, factorSigma=4):
  '''resp: 2D array charactering the response'''
  tensor = computeTensor(im, sigmaG, factorSigma)
  resp = np.zeros((im.shape[0], im.shape[1]))
  for y, x in imIter(tensor):
      M = np.array([[tensor[y,x,0], tensor[y,x,1]],[tensor[y,x,1], tensor[y,x,2]]])
      resp[y,x] = np.linalg.det(M) - k*np.square(np.trace(M))
  return resp

def HarrisCorners(im, k=0.15, sigmaG=1, factor=4, maxiDiam=7, boundarySize=5):
  '''result: a list of points that locate the images' corners'''
  result = []
  resp = cornerResponse(im, k, sigmaG, factor)
  maxResp = ndimage.filters.maximum_filter(resp, maxiDiam)
  for y in xrange(boundarySize, (resp.shape[0]-boundarySize)):
      for x in xrange(boundarySize, (resp.shape[1]-boundarySize)):
          if resp[y,x] > 0:
              if resp[y,x] == maxResp[y,x]:
                  result.append(point(x, y))
  return result

def computeFeatures(im, cornerL, sigmaBlurDescriptor=0.5, radiusDescriptor=4):
  '''f_list: a list of feature objects'''
  f_list = []
  im_lum = BW(im)
  blurredIm = ndimage.filters.gaussian_filter(im_lum, [sigmaBlurDescriptor, sigmaBlurDescriptor])
  for P in cornerL:
      patch = descriptor(blurredIm, P, radiusDescriptor)
      f_list.append(feature(P, patch))
  return f_list
  
  #features=map(descriptor, im, cornerL)

def descriptor(blurredIm, P, radiusDescriptor=4):
  '''patch: descriptor around 2-D point P, with size (2*radiusDescriptor+1)^2 in 1-D'''
  patch = blurredIm[P.y-radiusDescriptor:P.y+radiusDescriptor+1, P.x-radiusDescriptor:P.x+radiusDescriptor+1]
  patch -= np.mean(patch)
  patch /= np.std(patch)
  return patch

def findCorrespondences(listFeatures1, listFeatures2, threshold=1.7):  
  '''correpondences: a list of correspondences object that associate two feature lists.'''
  correspondences = []
  comparison = [0]*len(listFeatures2)
  for feature in listFeatures1:
      minDist = float("inf")
      minPoint = None
      minDist2 = float("inf")
      for i in range(len(listFeatures2)):
          feature2 = listFeatures2[i]
          squareDist = np.sum(np.square(feature.descriptor - feature2.descriptor))
          if squareDist < minDist:
              minDist2 = minDist
              minDist = squareDist
              minPoint = feature2.pt
      if minDist2/minDist > threshold**2:
          correspondences.append(correspondence(feature.pt, minPoint))
  return correspondences

def RANSAC(listOfCorrespondences, Niter=1000, epsilon=4, acceptableProbFailure=1e-9):
  '''H_best: the best estimation of homorgraphy (3-by-3 matrix)'''
  '''inliers: A list of booleans that describe whether the element in listOfCorrespondences 
  an inlier or not'''
  ''' 6.815 can bypass acceptableProbFailure'''
  H_best = np.identity(3)
  inliers_best = []
  inliers_best_count = 0
  for i in xrange(Niter):
      X = float(inliers_best_count)/len(listOfCorrespondences)
      prob = 1 - (1 - X**4)**i
      if prob < acceptableProbFailure and i>0:
        break
      corrs = rnd.sample(listOfCorrespondences, 4)
      H = computeHomography(corrs)
      inlier_dist = map(lambda pair: eucDist(np.array([pair.pt2.y, pair.pt2.x]),applyHomAndGetEucCoors(H, pair.pt1)), listOfCorrespondences)
      inlier = map(lambda x: x<epsilon, inlier_dist)
      inlier_count = len(filter(lambda x: x==True, inlier))
      if inlier_count > inliers_best_count:
        inliers_best = inlier
        inliers_best_count = inlier_count
        H_best = H
  return (H_best, inliers_best)

def computeNHomographies(listOfListOfPairs, refIndex, blurDescriptior=0.5, radiusDescriptor=4):
  '''H_list: a list of Homorgraphy relative to L[refIndex]'''
  '''Note: len(H_list) is equal to len(listOfListOfPairs)'''
  Hlist = []
  for listOfPairs in listOfListOfPairs:
      H, inliers = RANSAC(listOfPairs)
      Hlist.append(H)
  Hlist.insert(refIndex, np.array([[1,0,0],[0,1,0],[0,0,1]]))
  for i in reversed(xrange(0, refIndex)):
      Hlist[i]=np.dot(Hlist[i+1],Hlist[i])
  for i in xrange((refIndex+1), len(Hlist)):
      Hlist[i]=np.dot(Hlist[i-1],np.linalg.inv(Hlist[i]))
  return Hlist

def autostitch(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  '''Use your a6 code to stitch the images. You need to hand in your A6 code'''
  f_lists = map(lambda im: computeFeatures(im, HarrisCorners(im), blurDescriptor, radiusDescriptor), L)
  listOfListOfPairs = []
  for i in xrange(1, len(f_lists)):
    correspondences = findCorrespondences(f_lists[i-1], f_lists[i])
    listOfListOfPairs.append(correspondences)
  H_list = computeNHomographies(listOfListOfPairs, refIndex, blurDescriptor, radiusDescriptor)
  return a6.compositeNImages(L, H_list)

def weight_map(h,w):
  ''' Given the image dimension h and w, return the hxwx3 weight map for linear blending'''
  w_map = np.zeros((h,w,3))
  half_h = float(h)/2
  half_w = float(w)/2
  for x in xrange(h):
    for y in xrange(w):
      weight = (1 - abs(x-half_h)/(half_h))*(1 - abs(y-half_w)/(half_w))
      w_map[x, y, :] = np.array([weight, weight, weight])
  return w_map

def linear_blending(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  ''' Return the stitching result with linear blending'''
  weight_maps = map(lambda im: weight_map(im.shape[0], im.shape[1]), L)
  f_lists = map(lambda im: computeFeatures(im, HarrisCorners(im), blurDescriptor, radiusDescriptor), L)
  listOfListOfPairs = []
  for i in xrange(1, len(f_lists)):
    correspondences = findCorrespondences(f_lists[i-1], f_lists[i])
    listOfListOfPairs.append(correspondences)
  H_list = computeNHomographies(listOfListOfPairs, refIndex, blurDescriptor, radiusDescriptor)
  weighted_L = [0]*len(weight_maps)
  for i in xrange(len(weight_maps)):
    weighted_L[i] = weight_maps[i]*L[i]
  composite = a6.compositeNImages(weighted_L, H_list, True)
  composite_weight = a6.compositeNImages(weight_maps, H_list, True)
  out = np.nan_to_num(composite/composite_weight)
  return out

def two_scale_blending(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  ''' Return the stitching result with two scale blending'''
  weight_maps = map(lambda im: weight_map(im.shape[0], im.shape[1]), L)
  f_lists = map(lambda im: computeFeatures(im, HarrisCorners(im), blurDescriptor, radiusDescriptor), L)
  listOfListOfPairs = []
  for i in xrange(1, len(f_lists)):
    correspondences = findCorrespondences(f_lists[i-1], f_lists[i])
    listOfListOfPairs.append(correspondences)
  H_list = computeNHomographies(listOfListOfPairs, refIndex, blurDescriptor, radiusDescriptor)
  
  low_freq = map(lambda im: ndimage.filters.gaussian_filter(im, [2, 2, 0]), L)
  high_freq = [a - b for a, b in zip(L, low_freq)]
  
  low_weighted_L = [0]*len(weight_maps)
  for i in xrange(len(weight_maps)):
    low_weighted_L[i] = weight_maps[i]*low_freq[i]
  low_composite = a6.compositeNImages(low_weighted_L, H_list, True)
  low_composite_weight = a6.compositeNImages(weight_maps, H_list, True)
  low_stitched = np.nan_to_num(low_composite/low_composite_weight)
  high_stitched = a6.compositeNImagesWeights(high_freq, H_list, weight_maps)
  return low_stitched + high_stitched

# Helpers, you may use the following scripts for convenience.
def A7PointToA6Point(a7_point):
  return np.array([a7_point.y, a7_point.x, 1.0], dtype=np.float64)


def A7PairsToA6Pairs(a7_pairs):
  A7pointList1=map(lambda pair: pair.pt1 ,a7_pairs)
  A6pointList1=map(A7PointToA6Point, A7pointList1)
  A7pointList2=map(lambda pair: pair.pt2 ,a7_pairs)
  A6pointList2=map(A7PointToA6Point, A7pointList2)
  return zip(A6pointList1, A6pointList2)

def applyHomAndGetEucCoors(H, pt):
  homCoors = np.dot(H, np.array([pt.y, pt.x, 1]))
  return np.round(np.array([homCoors[0]/homCoors[2], homCoors[1]/homCoors[2]]))

def eucDist(pt1, pt2):
  return np.sum(np.square(pt1-pt2))

def addConstraint(systm, i, constr):
  '''Adds the constraint constr to the system of equations systm. constr is simply listOfPairs[i] from the argument to computeHomography. This function should fill in 2 rows of systm. We want the solution to our system to give us the elements of a homography that maps constr[0] to constr[1]. Does not return anything'''
  systm[i*2] = np.array([constr[0][0], constr[0][1], 1, 0, 0, 0, -constr[0][0]*constr[1][0], -constr[0][1]*constr[1][0], -constr[1][0]])
  systm[i*2+1] = np.array([0, 0, 0, constr[0][0], constr[0][1], 1, -constr[0][0]*constr[1][1], -constr[0][1]*constr[1][1], -constr[1][1]])

def computeHomography(A7listOfPairs):
  '''Computes and returns the homography that warps points listOfPairs[-][0] to listOfPairs[-][1]'''
  listOfPairs = A7PairsToA6Pairs(A7listOfPairs)
  system = np.zeros((9,9), dtype=np.float64)
  system[8,8] = 1
  for i in range(4):
    addConstraint(system, i, listOfPairs[i])
  try:
    invSys = np.linalg.inv(system)
  except np.linalg.linalg.LinAlgError:
    invSys = np.identity(9)
  B = np.zeros(9)
  B[8] = 1
  out = np.dot(invSys, B)
  return np.reshape(out, (3,3))

def BW(im, weights=[0.3,0.6,0.1]):
    img = im.copy()
    im_out = np.zeros((img.shape[0], img.shape[1]))
    im_out =  np.dot(img[:,:],weights)
    return im_out
    
def gradientXY(im):
    ''' Return the sum of the absolute value of the gradient  
    The gradient is the filtered image by Sobel filter '''
    Sobel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    imGH = ndimage.filters.convolve(im, Sobel, mode='reflect')
    imGV = ndimage.filters.convolve(im, Sobel.transpose(), mode='reflect')
    return (imGH, imGV)

def imIter(im, debug=False, lim=1e6):
    for y in xrange(min(im.shape[0], lim)):
        if debug & (y%10==0): print 'y=', y
        for x in xrange(min(lim, im.shape[1])):
            yield y, x

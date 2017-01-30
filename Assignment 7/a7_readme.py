def time_spent():
  '''N: # of hours you spent on this one'''
  return 12

def collaborators():
  '''Eg. ppl=['batman', 'ninja'] (use their athena username)'''
  return ['none']

def potential_issues():
  return "my panorma is the wrong orientation because when I tried to rotate the images in preview it added in an alpha channel and messed everything up.  I figured this sideways image was still fine since it still stitches the images appropriately."

def extra_credit():
#```` Return the function names you implemended````
#```` Eg. return ['full_sift', 'bundle_adjustment']````
  return ['none']


def most_exciting():
  return a_string_with_explanation

def most_difficult():
  return a_string_with_explanation

def my_panorama():
  input_images=['my-photo-1.png', 'my-photo-2.png', 'my-photo-3.png']
  output_images=['myPanorama.png', 'myLinearBlendingPano.png','myTwoScaleBlendingPano.png']
  return ([input_images], output_images)

def my_debug():
  '''return (1) a string explaining how you debug
  (2) the images you used in debugging.

  Eg
  images=['debug1.jpg', 'debug2jpg']
  my_debug='I used blahblahblah...
  '''
  return ('the hardest part to debug for me was the linear blending.  I needed to print out the intermediate pieces along the way.  I started by printing out the results of the weight map function to ensure that was being produced acurately. I then printed out my results from my composite image that had been skewed by the weights since my image was appearing way too bright.  I also printed out the morphed weight image.  This showed me that I wasnt actually multiplying by the weights acurately and I was able to solve the problem', ['weightmap.png', 'composite.png', 'composite_weight.png'])

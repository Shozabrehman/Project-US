
import numpy as np 
import matplotlib.pyplot as plt


import imageio

from scipy.stats import pearsonr

im = np.array(imageio.imread("usimage.png"),'float')

refImage = im[5:,:]
curImage = im[:-5,:]







w2 = 5
h2 = 30

c = (140,100)

region = slice(c[0]-h2,c[0]+h2),slice(c[1]-w2,c[1]+w2)

referencePatch = refImage[region]

if(1):
  fig,ax = plt.subplots(3,figsize = [15,9])
  ax[0].imshow(refImage)
  ax[1].imshow(curImage)
  ax[2].imshow(referencePatch)
  plt.show()



if():
  im2 = im.copy()
  im2[region] = im2[region]*2
  plt.imshow(im2)


  

czRange = np.arange(100,200)

measure = np.zeros(czRange.shape)

for i,cz in enumerate(czRange):
  candidateRegion = slice(cz-h2,cz+h2),slice(c[1]-w2,c[1]+w2)
  candidatePatch = curImage[candidateRegion]
  
  
  # pearson's correlation
  measure[i] = pearsonr(referencePatch.reshape(referencePatch.size),candidatePatch.reshape(candidatePatch.size))[0]
  
  
  # SSD measure
  #measure[i] = np.sum((patch-candidatePatch)**2)
  
  
  
  plt.show()
  if(1):
    im2 = im.copy()
    im2[candidateRegion] = im2[candidateRegion]*2
    plt.imshow(im2)
    plt.show()
  

result = czRange[np.argmax(measure)]
print(result)
plt.plot(czRange,measure)
plt.xlabel("z")







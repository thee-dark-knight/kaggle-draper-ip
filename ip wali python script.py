# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:07:13 2016

@author: ASUS-1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
smjpegs = [f for f in glob.glob("../input/train_sm/*.jpeg")]
print(smjpegs[:9])


set175 = [smj for smj in smjpegs if "set175" in smj]
print(set175)


#Basic Exploration

first = plt.imread('../input/train_sm/set175_1.jpeg')
dims = np.shape(first)
print(dims)

np.min(first), np.max(first)


pixel_matrix = np.reshape(first, (dims[0] * dims[1], dims[2]))
print(np.shape(pixel_matrix))


#plt.scatter(pixel_matrix[:,0], pixel_matrix[:,1])
_ = plt.hist2d(pixel_matrix[:,1], pixel_matrix[:,2], bins=(50,50))




fifth = plt.imread('../input/train_sm/set175_5.jpeg')
dims = np.shape(fifth)
pixel_matrix5 = np.reshape(fifth, (dims[0] * dims[1], dims[2]))


_ = plt.hist2d(pixel_matrix5[:,1], pixel_matrix5[:,2], bins=(50,50))


_ = plt.hist2d(pixel_matrix[:,2], pixel_matrix5[:,2], bins=(50,50))



plt.imshow(first)
plt.imshow(fifth)



plt.imshow(first[:,:,2] - fifth[:,:,1])


second = plt.imread('../input/train_sm/set175_2.jpeg')
plt.imshow(first[:,:,2] - second[:,:,2])


plt.imshow(second)




# simple k means clustering
from sklearn import cluster

kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix)

dims = np.shape(first)
clustered_img = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img)


plt.imshow(first)

#doesn't run on server
ind0, ind1, ind2, ind3 = [np.where(clustered == x)[0] for x in [0, 1, 2, 3]]


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_vals = [('r', 'o', ind0),
             ('b', '^', ind1),
             ('g', '8', ind2),
             ('m', '*', ind3)]

for c, m, ind in plot_vals:
    xs = pixel_matrix[ind, 0]
    ys = pixel_matrix[ind, 1]
    zs = pixel_matrix[ind, 2]
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('Blue channel')
ax.set_ylabel('green channel')
ax.set_zlabel('Red channel')




# quick look at color value histograms for pixel matrix from first image
import seaborn as sns
sns.distplot(pixel_matrix[:,0], bins=12)
sns.distplot(pixel_matrix[:,1], bins=12)
sns.distplot(pixel_matrix[:,2], bins=12)


set79 = [smj for smj in smjpegs if "set79" in smj]
print(set79)

img79_1, img79_2, img79_3, img79_4, img79_5 = \
  [plt.imread("../input/train_sm/set79_" + str(n) + ".jpeg") for n in range(1, 6)]
  
  
  img_list = (img79_1, img79_2, img79_3, img79_4, img79_5)

plt.figure(figsize=(8,10))
plt.imshow(img_list[0])
plt.show()




class MSImage():
    """Lightweight wrapper for handling image to matrix transforms. No setters,
    main point of class is to remember image dimensions despite transforms."""
    
    def __init__(self, img):
        """Assume color channel interleave that holds true for this set."""
        self.img = img
        self.dims = np.shape(img)
        self.mat = np.reshape(img, (self.dims[0] * self.dims[1], self.dims[2]))

    @property
    def matrix(self):
        return self.mat
        
    @property
    def image(self):
        return self.img
    
    def to_flat_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form when
        derived image would only have one band."""
        return np.reshape(derived, (self.dims[0], self.dims[1]))
    
    def to_matched_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form."""
        return np.reshape(derived, (self.dims[0], self.dims[1], self.dims[2]))
        
        
        
        
        
        
msi79_1 = MSImage(img79_1)
print(np.shape(msi79_1.matrix))
print(np.shape(msi79_1.img))



def bnormalize(mat):
    """much faster brightness normalization, since it's all vectorized"""
    bnorm = np.zeros_like(mat, dtype=np.float32)
    maxes = np.max(mat, axis=1)
    bnorm = mat / np.vstack((maxes, maxes, maxes)).T
    return bnorm
    
    
    
    
bnorm = bnormalize(msi79_1.matrix)
bnorm_img = msi79_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()




msi79_2 = MSImage(img79_2)
bnorm79_2 = bnormalize(msi79_2.matrix)
bnorm79_2_img = msi79_2.to_matched_img(bnorm79_2)
plt.figure(figsize=(8,10))
plt.imshow(bnorm79_2_img)
plt.show()


msinorm79_1 = MSImage(bnorm_img)
msinorm79_2 = MSImage(bnorm79_2_img)

_ = plt.hist2d(msinorm79_1.matrix[:,2], msinorm79_2.matrix[:,2], bins=(50,50))


_ = plt.hist2d(msinorm79_1.matrix[:,1], msinorm79_2.matrix[:,1], bins=(50,50))


_ = plt.hist2d(msinorm79_1.matrix[:,0], msinorm79_2.matrix[:,0], bins=(50,50))


import seaborn as sns
sns.distplot(msinorm79_1.matrix[:,0], bins=12)
sns.distplot(msinorm79_1.matrix[:,1], bins=12)
sns.distplot(msinorm79_1.matrix[:,2], bins=12)



plt.figure(figsize=(8,10))
plt.imshow(img79_1)
plt.show()


np.max(img79_1[:,:,0])


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(img79_1[:,:,0] > 230)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()



plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(img79_2[:,:,0] > 230)
plt.subplot(122)
plt.imshow(img79_2)
plt.show()



print(np.min(bnorm79_2_img[:,:,0]))
print(np.max(bnorm79_2_img[:,:,0]))
print(np.mean(bnorm79_2_img[:,:,0]))
print(np.std(bnorm79_2_img[:,:,0]))


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm79_2_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(img79_2)
plt.show()


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()




plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow((bnorm79_2_img[:,:,0] > 0.9999) & \
           (bnorm79_2_img[:,:,1] < 0.9999) & \
           (bnorm79_2_img[:,:,2] < 0.9999))
plt.subplot(122)
plt.imshow(img79_2)
plt.show()


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.995)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()


plt.figure(figsize=(10,6))
plt.subplot(121)
plt.plot(bnorm_img[2000, 1000, :])
plt.subplot(122)
plt.plot(img79_1[2000, 1000, :])



from scipy import spatial

pixel = msi79_1.matrix[2000 * 1000, :]
np.shape(pixel)








def spectral_angle_mapper(pixel):
    return lambda p2: spatial.distance.cosine(pixel, p2)

match_pixel = np.apply_along_axis(spectral_angle_mapper(pixel), 1, msi79_1.matrix)

plt.figure(figsize=(10,6))
plt.imshow(msi79_1.to_flat_img(match_pixel < 0.0000001))

def summary(mat):
    print("Max: ", np.max(mat),
          "Min: ", np.min(mat),
          "Std: ", np.std(mat),
          "Mean: ", np.mean(mat))

summary(match_pixel)









set144 = [MSImage(plt.imread(smj)) for smj in smjpegs if "set144" in smj]

plt.imshow(set144[0].image)


import skimage
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import sobel


# a sobel filter is a basic way to get an edge magnitude/gradient image
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel(set144[0].image[:750,:750,2]))



from skimage.filters import sobel_h

# can also apply sobel only across one direction.
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel_h(set144[0].image[:750,:750,2]), cmap='BuGn')



from sklearn.decomposition import PCA

pca = PCA(3)
pca.fit(set144[0].matrix)
set144_0_pca = pca.transform(set144[0].matrix)
set144_0_pca_img = set144[0].to_matched_img(set144_0_pca)


fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,0], cmap='BuGn')


fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,1], cmap='BuGn')


fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,2], cmap='BuGn')


sub = set144[0].image[:150,:150,2]


def glcm_image(img, measure="dissimilarity"):
    """TODO: allow different window sizes by parameterizing 3, 4. Also should
    parameterize direction vector [1] [0]"""
    texture = np.zeros_like(sub)

    # quadratic looping in python w/o vectorized routine, yuck!
    for i in range(img.shape[0] ):  
        for j in range(sub.shape[1] ):  
          
            # don't calculate at edges
            if (i < 3) or \
               (i > (img.shape[0])) or \
               (j < 3) or \
               (j > (img.shape[0] - 4)):          
                continue  
        
            # calculate glcm matrix for 7 x 7 window, use dissimilarity (can swap in
            # contrast, etc.)
            glcm_window = img[i-3: i+4, j-3 : j+4]  
            glcm = greycomatrix(glcm_window, [1], [0],  symmetric = True, normed = True )   
            texture[i,j] = greycoprops(glcm, measure)  
    return texture
    
    
dissimilarity = glcm_image(sub, "dissimilarity")
    




fig = plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(dissimilarity, cmap="bone")
plt.subplot(1,2,2)
plt.imshow(sub, cmap="bone")




from skimage import color

hsv = color.rgb2hsv(set144[0].image)


fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(set144[0].image, cmap="bone")
plt.subplot(2,2,2)
plt.imshow(hsv[:,:,0], cmap="bone")
plt.subplot(2,2,3)
plt.imshow(hsv[:,:,1], cmap='bone')
plt.subplot(2,2,4)
plt.imshow(hsv[:,:,2], cmap='bone')




fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(set144[0].image[:200,:200,:])
plt.subplot(2,2,2)
plt.imshow(hsv[:200,:200,0], cmap="PuBuGn")
plt.subplot(2,2,3)
plt.imshow(hsv[:200,:200,1], cmap='bone')
plt.subplot(2,2,4)
plt.imshow(hsv[:200,:200,2], cmap='bone')




fig = plt.figure(figsize=(8, 6))
plt.imshow(hsv[200:500,200:500,0], cmap='bone')


hsvmsi = MSImage(hsv)


import seaborn as sns
sns.distplot(hsvmsi.matrix[:,0], bins=12)
sns.distplot(hsvmsi.matrix[:,1], bins=12)
sns.distplot(hsvmsi.matrix[:,2], bins=12)



#Damn understanding this is gonna be cool!





#2





########################################################################
#
# Taking the1owls Image Matching script and experimenting;
# - downsizing images to speed up
#
# Want to extract the warp parameters
#
########################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt

print(cv2.__version__)


def im_stitcher(imp1, imp2, pcntDownsize = 1.0, withTransparency=False):
    
    #Read image1
    image1 = cv2.imread(imp1)
    
    # perform the resizing of the image by pcntDownsize and create a Grayscale version
    dim1 = (int(image1.shape[1] * pcntDownsize), int(image1.shape[0] * pcntDownsize))
    img1 = cv2.resize(image1, dim1, interpolation = cv2.INTER_AREA)
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    #Read image2
    image2 = cv2.imread(imp2)
    
    # perform the resizing of the image by pcntDownsize and create a Grayscale version
    dim2 = (int(image2.shape[1] * pcntDownsize), int(image2.shape[0] * pcntDownsize))
    img2 = cv2.resize(image2, dim2, interpolation = cv2.INTER_AREA)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    #use BRISK to create keypoints in each image
    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img1Gray,None)
    kp2, des2 = brisk.detectAndCompute(img2Gray,None)
    
    # use BruteForce algorithm to detect matches among image keypoints 
    dm = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    
    matches = dm.knnMatch(des1,des2, 2)
    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches_.append((m[0].trainIdx, m[0].queryIdx))
    
    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1,1,2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1,1,2)
    
    
    H, mask = cv2.findHomography(kp2_,kp1_, cv2.RANSAC, 4.0)
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    
    t = [-xmin,-ymin]
    
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    
    #warp the colour version of image2
    im = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
    #overlay colur version of image1 to warped image2
    if withTransparency == True:
        h3,w3 = im.shape[:2]
        bim = np.zeros((h3,w3,3), np.uint8)
        bim[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
        
        #imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #imColor = cv2.applyColorMap(imGray, cv2.COLORMAP_JET)
        
        #im =(im[:,:,2] - bim[:,:,2])
        im = cv2.addWeighted(im,0.6,bim,0.6,0)
    else:
        im[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return(im)

##########################################################
#
# Match all combinations of one set of images
#
##########################################################

#img104_1_166_1 = im_stitcher("../input/test_sm/set104_1.jpeg", "../input/test_sm/set166_1.jpeg", 0.4, True)
#img104_2_166_4 = im_stitcher("../input/test_sm/set104_2.jpeg", "../input/test_sm/set166_4.jpeg", 0.4, True)
#img104_3_166_5 = im_stitcher("../input/test_sm/set104_3.jpeg", "../input/test_sm/set166_5.jpeg", 0.4, True)
#img104_4_166_3 = im_stitcher("../input/test_sm/set104_4.jpeg", "../input/test_sm/set166_3.jpeg", 0.4, True)
#img104_5_166_2 = im_stitcher("../input/test_sm/set104_5.jpeg", "../input/test_sm/set166_2.jpeg", 0.4, True)

#plt.imsave('Set104_1_166_1_BRISK_matching.jpeg',img104_1_166_1) 
#plt.imsave('Set104_2_166_4_BRISK_matching.jpeg',img104_2_166_4) 
#plt.imsave('Set104_3_166_5_BRISK_matching.jpeg',img104_3_166_5) 
#plt.imsave('Set104_4_166_3_BRISK_matching.jpeg',img104_4_166_3) 
#plt.imsave('Set104_6_166_2_BRISK_matching.jpeg',img104_5_166_2) 

#img1_1_85_5 = im_stitcher("../input/test_sm/set1_1.jpeg", "../input/test_sm/set85_5.jpeg", 0.4, True)
#img1_2_85_4 = im_stitcher("../input/test_sm/set1_2.jpeg", "../input/test_sm/set85_4.jpeg", 0.4, True)
#img1_3_85_2 = im_stitcher("../input/test_sm/set1_3.jpeg", "../input/test_sm/set85_2.jpeg", 0.4, True)
#img1_4_85_3 = im_stitcher("../input/test_sm/set1_4.jpeg", "../input/test_sm/set85_3.jpeg", 0.4, True)
#img1_5_85_1 = im_stitcher("../input/test_sm/set1_5.jpeg", "../input/test_sm/set85_1.jpeg", 0.4, True)

#plt.imsave('Set1_1_85_5_BRISK_matching.jpeg',img1_1_85_5) 
#plt.imsave('Set1_2_85_4_BRISK_matching.jpeg',img1_2_85_4) 
#plt.imsave('Set1_3_85_2_BRISK_matching.jpeg',img1_3_85_2) 
#plt.imsave('Set1_4_85_3_BRISK_matching.jpeg',img1_4_85_3) 
#plt.imsave('Set1_5_85_1_BRISK_matching.jpeg',img1_5_85_1) 

#img3_1_22_1 = im_stitcher("../input/test_sm/set3_1.jpeg", "../input/test_sm/set22_1.jpeg", 0.4, True)
#img3_2_22_2 = im_stitcher("../input/test_sm/set3_2.jpeg", "../input/test_sm/set22_2.jpeg", 0.4, True)
#img3_3_22_5 = im_stitcher("../input/test_sm/set3_3.jpeg", "../input/test_sm/set22_5.jpeg", 0.4, True)
#img3_4_22_3 = im_stitcher("../input/test_sm/set3_4.jpeg", "../input/test_sm/set22_3.jpeg", 0.4, True)
#img3_5_22_4 = im_stitcher("../input/test_sm/set3_5.jpeg", "../input/test_sm/set22_4.jpeg", 0.4, True)

#plt.imsave('Set3_1_22_1_BRISK_matching.jpeg',img3_1_22_1) 
#plt.imsave('Set3_2_22_2_BRISK_matching.jpeg',img3_2_22_2) 
#plt.imsave('Set3_3_22_5_BRISK_matching.jpeg',img3_3_22_5) 
#plt.imsave('Set3_4_22_3_BRISK_matching.jpeg',img3_4_22_3) 
#plt.imsave('Set3_5_22_4_BRISK_matching.jpeg',img3_5_22_4) 

#img5_1_68_3 = im_stitcher("../input/train_sm/set5_1.jpeg", "../input/test_sm/set68_3.jpeg", 0.4, True)
#plt.imsave('Set5_1_68_3_BRISK_matching.jpeg',img5_1_68_3) 

img160_5_74_1 = im_stitcher("../input/train_sm/set160_5.jpeg", "../input/test_sm/set74_1.jpeg", 1, True)
img160_5_74_2 = im_stitcher("../input/train_sm/set160_5.jpeg", "../input/test_sm/set74_2.jpeg", 0.4, True)
img160_5_74_3 = im_stitcher("../input/train_sm/set160_5.jpeg", "../input/test_sm/set74_3.jpeg", 0.4, True)
img160_5_74_4 = im_stitcher("../input/train_sm/set160_5.jpeg", "../input/test_sm/set74_4.jpeg", 0.4, True)
img160_5_74_5 = im_stitcher("../input/train_sm/set160_5.jpeg", "../input/test_sm/set74_5.jpeg", 0.4, True)

plt.imsave('Set160_5_74_1_BRISK_matching.jpeg',img160_5_74_1) 
plt.imsave('Set160_5_74_2_BRISK_matching.jpeg',img160_5_74_2) 
plt.imsave('Set160_5_74_3_BRISK_matching.jpeg',img160_5_74_3) 
plt.imsave('Set160_5_74_4_BRISK_matching.jpeg',img160_5_74_4) 
plt.imsave('Set160_5_74_5_BRISK_matching.jpeg',img160_5_74_5) 

















#3







import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from PIL import ImageFilter
import multiprocessing
import random; random.seed(2016);
import cv2
import re
import os, glob

sample_sub = pd.read_csv('../input/sample_submission.csv')
train_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/train_sm/*.jpeg")])
train_files.columns = ['path', 'group', 'pic_no']
test_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/test_sm/*.jpeg")])
test_files.columns = ['path', 'group', 'pic_no']
print(len(train_files),len(test_files),len(sample_sub))
train_images = train_files[train_files["group"]=='set107']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.subplots_adjust(wspace=0, hspace=0)
i_ = 0
a = []
for l in train_images.path:
    im = cv2.imread(l)
    plt.subplot(5, 2, i_+1).set_title(l)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(5, 2, i_+2).set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)







kaze = cv2.KAZE_create()
akaze = cv2.AKAZE_create()
brisk = cv2.BRISK_create()

plt.rcParams['figure.figsize'] = (7.0, 18.0)
plt.subplots_adjust(wspace=0, hspace=0)
i = 0
for detector in [kaze, akaze, brisk]:
    start_time = time.time()
    im = cv2.imread(train_images.path[0])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    (kps, descs) = detector.detectAndCompute(gray, None)       
    cv2.drawKeypoints(im, kps, im, (0, 255, 0))
    plt.subplot(3, 1, i+1).set_title(list(['kaze','akaze','brisk'])[i] + " " + str(round(((time.time() - start_time)/60),5)))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i+=1
    
    
    print(cv2.__version__)

img1 = cv2.imread(train_images.path[0], 0)
img2 = cv2.imread(train_images.path[1], 0)
brisk = cv2.BRISK_create()
kp1, des1 = brisk.detectAndCompute(img1,None)
kp2, des2 = brisk.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img1 = cv2.imread(train_images.path[0])
img2 = cv2.imread(train_images.path[1])
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100], flags=2, outImg=img2, matchColor = (0,255,0))
plt.rcParams['figure.figsize'] = (14.0, 8.0)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)); plt.axis('off')




brisk = cv2.BRISK_create()
dm = cv2.DescriptorMatcher_create("BruteForce")

def c_resize(img, ratio):
    wh = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
    img = cv2.resize(img, wh, interpolation = cv2.INTER_AREA)
    return img
    
def im_stitcher(imp1, imp2, imsr = 1.0, withTransparency=False):
    img1 = cv2.imread(imp1, 0)
    img2 = cv2.imread(imp2, 0)
    if imsr < 1.0:
        img1 = c_resize(img1,imsr); img2 = c_resize(img2,imsr)
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    kp1, des1 = brisk.detectAndCompute(img1,None)
    kp2, des2 = brisk.detectAndCompute(img2,None)
    matches = dm.knnMatch(des1,des2, 2)
    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches_.append((m[0].trainIdx, m[0].queryIdx))
    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1,1,2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1,1,2)
    H, mask = cv2.findHomography(kp2_,kp1_, cv2.RANSAC, 4.0)
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    img1 = cv2.imread(imp1)
    img2 = cv2.imread(imp2)
    if imsr < 1.0:
        img1 = c_resize(img1,imsr); img2 = c_resize(img2,imsr)
    im = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    if withTransparency == True:
        h3,w3 = im.shape[:2]
        bim = np.zeros((h3,w3,3), np.uint8)
        bim[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
        im = cv2.addWeighted(im,1.0,bim,0.9,0)
    else:
        im[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return im
    
    
    
img = im_stitcher(train_images.path[0], train_images.path[4], 0.5, True)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
img[np.where((img < [20,20,20]).all(axis = 2))] = [255,255,255]
plt.imshow(img); plt.axis('off')



img = cv2.imread(train_images.path[0])
cv2.imwrite('panoramic.jpeg',img)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
for i in range(1,5):
    img = im_stitcher(train_images.path[i], 'panoramic.jpeg', 0.5, False)
    cv2.imwrite('panoramic.jpeg',img)
img[np.where((img < [20,20,20]).all(axis = 2))] = [255,255,255]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')





train_images = train_files[train_files["group"]=='set4']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)
img = cv2.imread(train_images.path[0])
cv2.imwrite('panoramic2.jpeg',img)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
for i in range(1,5):
    img = im_stitcher(train_images.path[i], 'panoramic2.jpeg', 0.5, False)
    cv2.imwrite('panoramic2.jpeg',img)
img[np.where((img < [20,20,20]).all(axis = 2))] = [255,255,255]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')














#4










# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
def im_align_orb(imp1, imp2, nf=10000):
    """
    :param imp1: image1 file path
    :param imp2: image2 file path
    :param nf: max number of ORB key points
    :return:  transformed image2, so that it can be aligned with image1
    """
    img1 = cv2.imread(imp1, 0)
    img2 = cv2.imread(imp2, 0)
    h2, w2 = img2.shape[:2]

    orb = cv2.ORB_create(nfeatures=nf, WTA_K=2)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors.
    matches = bf.knnMatch(des1, des2, 2)

    # Sort them in the order of their distance.
    # matches_ = sorted(matches, key=lambda x: x.distance)[:5000]
    # print([m.distance for m in matches_])

    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches_.append((m[0].trainIdx, m[0].queryIdx))

    #print("len(kp1), len(kp2), len(matches_)")

    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1, 1, 2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(kp2_, kp1_, cv2.RANSAC, 1.0)

    h1, w1 = img1.shape[:2]

    img2 = cv2.warpPerspective(cv2.imread(imp2), H, (w1, h1))
    return img2

def align_set_by_id(setid, isTrain=True, nFeatures=20000):
    """
    :param setid:
    :param isTrain:
    :return:
    """
    train_path = '../input/train_sm/'
    test_path = '../input/test_sm/'

    if isTrain == True:
        image_path = train_path
        fn1 = train_path + "set" + str(setid) + "_1.jpeg"
        outputpath = "./train_output"
    else:
        image_path = test_path
        fn1 = test_path + "set" + str(setid) + "_1.jpeg"
        outputpath = "./test_output/" 
    
    result=list()
    result.append(cv2.cvtColor(cv2.imread(fn1), cv2.COLOR_BGR2RGB))
    for id in [2, 3, 4, 5]:
        fn2 = image_path + "set" + str(setid) + "_" + str(id) + ".jpeg"
        print("fn1=%s, fn2=%s" % (os.path.basename(fn1), os.path.basename(fn2)))
        im = im_align_orb(fn1, fn2, nFeatures)
        #Note: kaggle script seems can't save output image? 
        #cv2.imwrite(outputpath + os.path.basename(fn2), im)
        result.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    #Note: kaggle script seems can't save output image? 
    #cv2.imwrite(outputpath + os.path.basename(fn1), cv2.imread(fn1))
    
    return result
    
    


setimages=align_set_by_id(4, nFeatures=15000)



from PIL import Image 

plt.rcParams['figure.figsize'] = (16.0,16.0)

plt.subplot(321).set_title('image1'), plt.imshow(setimages[0]),plt.axis('off')
plt.subplot(323).set_title('image2'), plt.imshow(setimages[1]),plt.axis('off')
plt.subplot(324).set_title('image3'), plt.imshow(setimages[2]),plt.axis('off')
plt.subplot(325).set_title('image4'), plt.imshow(setimages[3]),plt.axis('off')
plt.subplot(326).set_title('image5'), plt.imshow(setimages[4]),plt.axis('off')

plt.show()




def align_all_set(path, isTrain=True):
    allfiles = os.listdir(path)
    allfiles = [os.path.basename(file) for file in allfiles if file.startswith('set')]
    allsets = np.unique([f.split("_")[0].replace("set", "") for f in allfiles])

    os.makedirs(path + "/output", exist_ok=True)

    for s in allsets:
        align_set_by_id(s, isTrain, nFeatures=20000)
        
        
        
        

    
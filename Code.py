
# coding: utf-8

# In[107]:


import numpy as np
import numpy.linalg
import os
import cv2
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[44]:


def getImgList(imgPath, imgFormat = 'jpg'):
   
    imgList = os.listdir(imgPath)
    imgList = [os.path.join(imgPath,imgName) for imgName in imgList if imgName.split('.')[-1] == imgFormat]
    return imgList

def getImgListPNG(imgPath, imgFormat = 'png'):
    
    imgList = os.listdir(imgPath)
    imgList = [os.path.join(imgPath,imgName) for imgName in imgList if imgName.split('.')[-1] == imgFormat]
    return imgList


# In[110]:


imgList_Y = getImgList('/Users/xuweiqi/Desktop/276A Project1/trainset/Yellow','jpg')
imgList_R = getImgList('/Users/xuweiqi/Desktop/276A Project1/trainset/Barrel Red','jpg')
imgList_NR = getImgList('/Users/xuweiqi/Desktop/276A Project1/trainset/NotBarrel Red','jpg')
imgList_B = getImgList('/Users/xuweiqi/Desktop/276A Project1/trainset/Brown','jpg')
imgList_test=getImgList（'/Users/xuweiqi/Desktop/276A Project1/testset','jpg'）



#print imgList_Y
#print imgList_test


# In[111]:


def getImgShape(imgName):
   
    return cv2.imread(imgName).shape


# In[112]:


## Test codes
imgShape = getImgShape(imgList_Y[0])
print imgShape


# In[113]:


def loadAllImgs(imgList):
    
    imgShape = getImgShape(imgList[0])
    allImgBGR = np.zeros((len(imgList),imgShape[0], imgShape[1], imgShape[2]),dtype=np.uint8) # 图像数据类型是无符号8位整型
    allImgHSV = np.zeros((len(imgList),imgShape[0], imgShape[1], imgShape[2]),dtype=np.uint8) #So cool you write this^
    for index, imgName in enumerate(imgList):
        img = cv2.imread(imgName)
        allImgBGR[index] = img
        allImgHSV[index] = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    return allImgBGR, allImgHSV 


# In[114]:


allImgBGR_Y, allImgHSV_Y = loadAllImgs(imgList_Y)
allImgBGR_R, allImgHSV_R = loadAllImgs(imgList_R)
allImgBGR_NR, allImgHSV_NR = loadAllImgs(imgList_NR)
allImgBGR_B, allImgHSV_B = loadAllImgs(imgList_B)
allImgBGR_test, allImgHSV_test = loadAllImgs(imgList_test)



## Test codes
#print allImgBGR.shape, allImgHSV.shape
print allImgBGR_test.dtype, allImgHSV_test.dtype
#convert into BGR then imshow
plt.figure (1)
plt.subplot(1,2,1)
#plt.imshow(cv2.cvtColor(allImgBGR_test[8],cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1,2,2)
#plt.imshow(cv2.cvtColor(allImgHSV_test[8],cv2.COLOR_HSV2RGB))
plt.axis('off')
plt.show()


# In[115]:




# In[116]:


pixel_select_Y = allImgHSV_Y[allImgHSV_Y[:,:,:,2] != 0]
pixel_select_R = allImgHSV_R[allImgHSV_R[:,:,:,2] != 0]
pixel_select_NR = allImgHSV_NR[allImgHSV_NR[:,:,:,2] != 0]
pixel_select_B = allImgHSV_B[allImgHSV_B[:,:,:,2] != 0]


# In[117]:


print pixel_select_Y.shape
#print pixel_select2[800]


# In[118]:


## compute mean and covariance
pixel_select_Y_mean = np.mean(pixel_select_Y, axis=0)  ## 按列
pixel_select_Y_mean = np.mat(pixel_select_Y_mean)
pixel_select_Y_cov = np.cov(pixel_select_Y.T)  ## 3*3协方差
pixel_select_Y_len=pixel_select_Y.shape[0]## 数量

pixel_select_R_mean = np.mean(pixel_select_R, axis=0)
pixel_select_R_mean = np.mat(pixel_select_R_mean)
pixel_select_R_cov = np.cov(pixel_select_R.T)
pixel_select_R_len=pixel_select_R.shape[0]

pixel_select_NR_mean = np.mean(pixel_select_NR, axis=0)
pixel_select_NR_mean = np.mat(pixel_select_NR_mean)
pixel_select_NR_cov = np.cov(pixel_select_NR.T)
pixel_select_NR_len=pixel_select_NR.shape[0]

pixel_select_B_mean = np.mean(pixel_select_B, axis=0)
pixel_select_B_mean = np.mat(pixel_select_B_mean)
pixel_select_B_cov = np.cov(pixel_select_B.T)
pixel_select_B_len=pixel_select_NR.shape[0]

print pixel_select_B_cov 


# In[119]:


pixel_select_Y_prior=float(pixel_select_Y_len)/(pixel_select_Y_len+pixel_select_R_len+pixel_select_NR_len+pixel_select_B_len)
pixel_select_R_prior=float(pixel_select_R_len)/(pixel_select_Y_len+pixel_select_R_len+pixel_select_NR_len+pixel_select_B_len)
pixel_select_NR_prior=float(pixel_select_NR_len)/(pixel_select_Y_len+pixel_select_R_len+pixel_select_NR_len+pixel_select_B_len)
pixel_select_B_prior=float(pixel_select_B_len)/(pixel_select_Y_len+pixel_select_R_len+pixel_select_NR_len+pixel_select_B_len)

print pixel_select_Y_prior
print pixel_select_R_prior
print pixel_select_NR_prior
print pixel_select_B_prior


# In[120]:
widtha = []
heighta = []
for x in range(1):
    img=allImgHSV_test[x]
    red1=[]
    red2=[]
    h, w, c = img.shape
    for i in range(h-1):
        for j in range(w-1):
            test_R=(np.mat(img[i,j])-pixel_select_R_mean)*np.linalg.inv(pixel_select_R_cov)*(np.mat(img[i,j])-pixel_select_R_mean).T+np.log(2*np.pi**3*np.linalg.det(pixel_select_R_cov))-2*np.log(pixel_select_R_prior)
            test_Y=(np.mat(img[i,j])-pixel_select_Y_mean)*np.linalg.inv(pixel_select_Y_cov)*(np.mat(img[i,j])-pixel_select_Y_mean).T+np.log(2*np.pi**3*np.linalg.det(pixel_select_Y_cov))-2*np.log(pixel_select_Y_prior)
            test_NR=(np.mat(img[i,j])-pixel_select_NR_mean)*np.linalg.inv(pixel_select_NR_cov)*(np.mat(img[i,j])-pixel_select_NR_mean).T+np.log(2*np.pi**3*np.linalg.det(pixel_select_NR_cov))-2*np.log(pixel_select_NR_prior)
            test_B=(np.mat(img[i,j])-pixel_select_B_mean)*np.linalg.inv(pixel_select_B_cov)*(np.mat(img[i,j])-pixel_select_B_mean).T+np.log(2*np.pi**3*np.linalg.det(pixel_select_B_cov))-2*np.log(pixel_select_B_prior)

            if test_R <= test_NR:
                if test_R <= test_Y:
                    if test_R <= test_B:
                        red1.append(i)
                        red2.append(j)

    len_red1 = len(red1) #the number of red pixels
    ######print len_red1

    


# In[122]:


    pic=np.zeros([h,w])    
    
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()
    
    for i in range(len_red1):
        pic[red1[i],red2[i]]=1

    plt.figure()
    plt.imshow(pic, cmap = 'gray')  
    plt.show()
    

    from skimage import data, util
    from skimage.measure import label, regionprops
    label_pic = label(pic)
    props = regionprops(label_pic)

    maxrow=[]
    minrow=[]
    maxcol=[]
    mincol=[]
    
    for prop in props:
        #print float(prop.filled_area)/float(w*h)
        if float(prop.filled_area)/float(w*h) > 0.0008 : #decide whether the selected area is a barrel or not
            (min_row, min_col, max_row, max_col) = prop.bbox #put all the boudary points of the selected areas in a list
            maxrow.append(max_row)
            minrow.append(min_row)
            maxcol.append(max_col)
            mincol.append(min_col)


    BottomLeftX=min(minrow)
    BottomLeftY=min(mincol)
    BottomRightX=max(maxrow)
    BottomRightY=max(maxcol)
    print BottomLeftX, BottomLeftY, BottomRightX, BottomRightY #get the four corners of the barrel
        
    plt.figure()
    plt.imshow(pic[min(minrow):max(maxrow), min(mincol):max(maxcol)], cmap = 'gray')
    plt.show()
    meature_of_wideth=[]
    meature_of_height=[]

    height=max(maxrow)-min(minrow)
    wideth=max(maxcol)-min(mincol)
    d_height=12.7422-0.0482*height
    d_wideth=13.1067-0.0757*wideth
    print wideth, height
    print d_wideth, d_height
    #measure_of_wideth.append(wideth)
    #measure_of_height.append(height)
    print wideth, height
    widtha.append( wideth) #get barrel wideth
    heighta.append( height) #get barrel height

      #  plt.figure()
      #  plt.imshow(pic[min_row:max_row, min_col:max_col], cmap = 'gray')
       # plt.show()
        


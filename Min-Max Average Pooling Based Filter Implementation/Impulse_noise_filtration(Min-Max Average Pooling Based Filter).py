from skimage import io,color,measure
import skimage
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from skimage.metrics import structural_similarity
from scipy import signal

def find_Nd(R):
  M = R.shape[0]
  N = R.shape[1]
  noise_mask  = np.ones([M,N])
  for i in range(M):
      for j in range(N):
          if R[i][j] == 0 or R[i][j] == 1:
              noise_mask[i][j]=0
  suma=0
  for i in range(M):
      for j in range(N):
          suma+= noise_mask[i][j]
  Nd = suma / (M *N)
  return Nd

def get_noise_free_window2(img,i,j):
    Wc2 = []
    Wc3 = []
    Wc = img[i-1:i+2,j-1:j+2].ravel()
   
    for k in Wc:
      if k!=0:
        Wc2.append(k)
    for y in Wc2:
      if y!=1:
        Wc3.append(y)
    
    return Wc3

def IEHCLND(img,Nd):
    img = np.pad(img,pad_width=1,mode='symmetric')
    alpha = math.floor(Nd/0.1)
    M = img.shape[0]
    N = img.shape[1]
    oImg = img
    for i in range(1,M-1):
        for j in range(1,N-1):
          if img[i][j] == 0 or img[i][j] ==1:
            Wc = get_noise_free_window2(img,i,j)
            if len(Wc)>alpha:
              oImg[i][j] = np.median(Wc)
    oImg = oImg[1:M-1,1:N-1]
          
    return oImg

def CMMP(I1,I2,Layer):
    I1 = np.pad(I1,pad_width=1,mode='symmetric')
    I2 = np.pad(I2,pad_width=1,mode='symmetric')
    M = I1.shape[0]
    N = I2.shape[1]
    O1 = I1
    O2 = I2
    for i in range(1,M-1):
        for j in range(1,N-1):
          if I1[i][j] == 0 or I1[i][j] ==1:
       
            Wc = get_noise_free_window2(I1,i,j)
            try:
              Wc.remove(1)
              Wc.remove(0)
            except:
              pass
          
            if len(Wc)>0:
              if Layer == 'Min':
                O1[i][j] = np.min(Wc)
              
                O2[i][j] = np.max(Wc)
              
              elif Layer == 'Max':
              
                O1[i][j] = np.max(Wc)
               
                O2[i][j] = np.min(Wc)
               
    O1 = O1[1:M-1,1:N-1]
    O2 = O2[1:M-1,1:N-1]    
    return O1,O2

def RandS(I1,I2,nImg):
   I1 = np.pad(I1,pad_width=1,mode='symmetric')
   I2 = np.pad(I2,pad_width=1,mode='symmetric')
   nImg = np.pad(nImg,pad_width=1,mode='symmetric')
   M = nImg.shape[0]
   N = nImg.shape[1]
   oImg = (I1+I2)/2
   for i in range(1,M-1):
     for j in range(1,N-1):
        Wc = []
        if nImg[i][j] ==0 or nImg[i][j] ==1:
          Wc = get_noise_free_window2(oImg,i,j)       
          
          oImg[i][j] = np.mean(Wc)
          
   oImg = oImg[1:M-1,1:N-1]
   
              
   return oImg

def aplica_filtru(nImg):
    Nd = find_Nd(nImg)  # calculam Noise density

    #verificam daca Nd depaseste pragul de 0.45 sau nu, daca nu, se va aplica transformarea conform functiei IEHCLND
    if Nd < 0.45:
        nImg = IEHCLND(nImg,Nd)
    else:
        nImg=nImg
    I1 = nImg
    I2 = nImg
    Layers = ['Max','Min','Min','Max']

    for Layer in Layers:
        I1,I2 = CMMP(I1,I2,Layer)

    OutImg = RandS(I1,I2,nImg)
    return OutImg

def mean_filter(img):
    med = signal.convolve2d(img,np.ones([3,3])/9,mode ='same',boundary='symm')
    return img

def get_images(imp_noise,gaus_noise):
  gaussian_noise_image_filtered = aplica_filtru(gaus_noise)
  impulse_noise_image_filtered = aplica_filtru(imp_noise)
  gausian_noise_median_filter = scipy.ndimage.median_filter(gaus_noise,(3,3))
  impulse_noise_median_filter = scipy.ndimage.median_filter(imp_noise,(3,3))
  gausian_noise_mean_filter = mean_filter(gaus_noise)
  impulse_noise_mean_filter = mean_filter(imp_noise)
  return (gaussian_noise_image_filtered, impulse_noise_image_filtered, gausian_noise_median_filter, impulse_noise_median_filter, gausian_noise_mean_filter, impulse_noise_mean_filter)

def calcul_MAE(img1,img2):
  M = img1.shape[0]
  N = img1.shape[1]
  MAE = 0
  for i in range(M):
    for j in range(N):
      MAE += np.float64(abs(np.float64(img1[i,j])-np.float64(img2[i,j])))
  return MAE/(M*N)

def calcul_SSIM(img1,img2):
  
  img1 = np.uint8(img1)

  img2 = np.uint8(img2*255)
  
  (score, diff) = structural_similarity(img1,img2,full=True,multichannel=True)
  return  str(score)

img = io.imread('img5.jpg')

#plt.figure(),plt.imshow(img,cmap='gray'),plt.show()

noise = 0.1
img_impulse_noise = skimage.util.random_noise(img,'s&p',amount = noise)

img_gaussian_noise = skimage.util.random_noise(img,'gaussian',var = noise)

#plt.figure(),plt.imshow(img),plt.colorbar(),plt.show()
R_imp = img_impulse_noise[:,:,0]
G_imp = img_impulse_noise[:,:,1]
B_imp = img_impulse_noise[:,:,2]

R_gau = img_gaussian_noise[:,:,0]
G_gau = img_gaussian_noise[:,:,1]
B_gau = img_gaussian_noise[:,:,2]


#plt.figure(),plt.imshow(img_impulse_noise),plt.colorbar(),plt.show()
#plt.figure(),plt.imshow(img_gaussian_noise),plt.colorbar(),plt.show()

images = get_images(R_imp,R_gau)
gaussian_noise_image_filtered_R = images[0]
impulse_noise_image_filtered_R = images[1]
gausian_noise_median_filter_R = images[2]
impulse_noise_median_filter_R = images[3]
gausian_noise_mean_filter_R = images[4]
impulse_noise_mean_filter_R = images[5]

images = get_images(G_imp,G_gau)
gaussian_noise_image_filtered_G = images[0]
impulse_noise_image_filtered_G = images[1]
gausian_noise_median_filter_G = images[2]
impulse_noise_median_filter_G = images[3]
gausian_noise_mean_filter_G = images[4]
impulse_noise_mean_filter_G = images[5]

images = get_images(B_imp,B_gau)
gaussian_noise_image_filtered_B = images[0]
impulse_noise_image_filtered_B = images[1]
gausian_noise_median_filter_B = images[2]
impulse_noise_median_filter_B = images[3]
gausian_noise_mean_filter_B = images[4]
impulse_noise_mean_filter_B = images[5]

def get_RGB(img,cR,cG,cB):
  M = np.shape(img)[0]
  N = np.shape(img)[1]
  imagine_filtrata = np.zeros([M,N,3])
  imagine_filtrata[:,:,0] = cR
  imagine_filtrata[:,:,1] = cG
  imagine_filtrata[:,:,2] = cB
  return imagine_filtrata

img_zgomot_gaussian_filtrata = get_RGB(img_gaussian_noise,gaussian_noise_image_filtered_R,gaussian_noise_image_filtered_G,gaussian_noise_image_filtered_B)
img_zgomot_impulsiv_filtrata = get_RGB(img_impulse_noise,impulse_noise_image_filtered_R,impulse_noise_image_filtered_G,impulse_noise_image_filtered_B)

img_zgomot_gaussian_filtrata_median = get_RGB(img_gaussian_noise,gausian_noise_median_filter_R,gausian_noise_median_filter_G,gausian_noise_median_filter_B)
img_zgomot_impulsiv_filtrata_median = get_RGB(img_impulse_noise,impulse_noise_median_filter_R,impulse_noise_median_filter_G,impulse_noise_median_filter_B)

img_zgomot_gaussian_filtrata_mean = get_RGB(img_gaussian_noise,gausian_noise_mean_filter_R,gausian_noise_mean_filter_G,gausian_noise_mean_filter_B)
img_zgomot_impulsiv_filtrata_mean = get_RGB(img_impulse_noise,impulse_noise_mean_filter_R,impulse_noise_mean_filter_G,impulse_noise_mean_filter_B)

fig = plt.figure(figsize=(500, 25))
rows = 3
columns = 3 
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(img)
plt.axis('off')
plt.title("Original Picture",fontsize =10)
  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(img_gaussian_noise)
plt.axis('off')
x = float(calcul_SSIM(img,img_gaussian_noise))
x = round(x,2)
plt.title("Imagine zgomot Gaussian Similaritate: " + str(x),fontsize =10)
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(img_zgomot_gaussian_filtrata)
plt.axis('off')
x = float(calcul_SSIM(img,img_zgomot_gaussian_filtrata))
x = round(x,2)
plt.title("Imagine zgomot Gaussian filtrata Similaritate: " + str(x),fontsize =10)
  
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
  
# showing image
plt.imshow(img_impulse_noise)
plt.axis('off')
x = float(calcul_SSIM(img,img_impulse_noise))
x = round(x,2)
plt.title("Imagine zgomot impulsiv Similaritate: " + str(x),fontsize =10)

# Adds a subplot at the 5th position
fig.add_subplot(rows, columns, 5)
  
# showing image
plt.imshow(img_zgomot_impulsiv_filtrata)
plt.axis('off')
x = float(calcul_SSIM(img,img_zgomot_impulsiv_filtrata))
x = round(x,2)
plt.title("Imagine zgomot impulsiv filtrata Similaritate: " + str(x),fontsize =10)

# Adds a subplot at the 5th position
fig.add_subplot(rows, columns, 6)
  
# showing image
plt.imshow(img_zgomot_gaussian_filtrata_median)
plt.axis('off')
x = float(calcul_SSIM(img,img_zgomot_gaussian_filtrata_median))
x = round(x,2)
plt.title("Imagine zgomot Gaussian median filter Similaritate: " + str(x),fontsize =10)

# Adds a subplot at the 5th position
fig.add_subplot(rows, columns, 7)
  
# showing image
plt.imshow(img_zgomot_impulsiv_filtrata_median)
plt.axis('off') 
x = float(calcul_SSIM(img,img_zgomot_impulsiv_filtrata_median))
x = round(x,2)
plt.title("Imagine zgomot impulsiv median filter Similaritate: " + str(x) ,fontsize =10)

# Adds a subplot at the 5th position
fig.add_subplot(rows, columns, 8)
  
# showing image
plt.imshow(img_zgomot_gaussian_filtrata_mean)
plt.axis('off')
x = float(calcul_SSIM(img,img_zgomot_gaussian_filtrata_mean))
x = round(x,2)
plt.title("Imagine zgomot Gaussian mean filter Similaritate: " + str(x),fontsize =10)

# Adds a subplot at the 5th position
fig.add_subplot(rows, columns, 9)
  
# showing image
plt.imshow(img_zgomot_impulsiv_filtrata_mean)
plt.axis('off')
x = float(calcul_SSIM(img,img_zgomot_impulsiv_filtrata_mean))
x = round(x,2)
plt.title("Imagine zgomot impulsiv mean filter Similaritate: " + str(x),fontsize =10)
plt.show()


print('Eroarea intre imaginea originala si cea cu zgomot este ' +str(calcul_MAE(img,img_impulse_noise)))
print('Eroarea intre imaginea originala si cea filtrata este '+ str(calcul_MAE(img,img_zgomot_impulsiv_filtrata)))
print('Eroarea intre imaginea originala si cea filtrata median este '+ str(calcul_MAE(img,img_zgomot_impulsiv_filtrata_median)))
print('Eroarea intre imaginea originala si cea filtrata mean este '+ str(calcul_MAE(img,img_zgomot_impulsiv_filtrata_mean)))

print('Similaritatea intre imaginea originala si cea cu zgomot este ' +str(calcul_SSIM(img,img_impulse_noise)))
print('Similaritatea intre imaginea originala si cea filtrata este '+ str(calcul_SSIM(img,img_zgomot_impulsiv_filtrata)))
print('Similaritatea intre imaginea originala si cea filtrata median este '+ str(calcul_SSIM(img,img_zgomot_impulsiv_filtrata_median)))
print('Similaritatea intre imaginea originala si cea filtrata mean este '+ str(calcul_SSIM(img,img_zgomot_impulsiv_filtrata_mean)))

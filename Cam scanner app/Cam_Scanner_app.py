import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
import math
from skimage import io

image = cv2.imread("test2.jpg")  #citim imaginea 
img_orig = image.copy() #cream o copie a imaginii originale

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #convertim imaginea in alb-negru
#plt.figure(),plt.imshow(grayscale,cmap='gray'),plt.colorbar(),plt.show()
blurred = cv2.blur(grayscale, (3,3)) #bluram imaginea cu ajutorul unui filtru de dim 3x3
edges = cv2.Canny(blurred,50,200) # extragem marginile cu ajutorul functiei Canny 50 si 200 sunt valori de treshhold pentru histerezis

#plt.figure(), plt.imshow(edges,cmap = "gray"),plt.show()
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #extragem contururile

c = max(contours, key = cv2.contourArea) #salvam intr-o variabila valoarea maxima a conturului care in cazul nostru ar trebui sa fie pagina scrisa 

perimeter = cv2.arcLength(c, True) #calculam lungimea perimetrului, care ar trebui sa fie un contur inchis

epsilon = 0.03 # 3%
approx = cv2.approxPolyDP(c, epsilon*perimeter, True) #aproximama forma conturului pentru a obtine o forma mai robusta, argumentul true se refera la o forma inchisa

contoured_image = cv2.drawContours(image, [approx], -1, (0,255,0), 2) #desenam conturul calculat peste imaginea originala

#plt.figure(), plt.imshow(contoured_image),plt.show()

approx = np.concatenate(approx,axis=0,dtype='float32') #transformam coordonatele intr o matrice 4x2
#print(approx)

def order_points(approx):  #ordonam punctele pentru a obtine coordonatele intr o anumita ordine

	coord = np.zeros((4, 2), dtype = "float32")
	
    # Coltul din stanga sus va avea cea mai mica suma iar cel din dreapta jos cea mai mare suma
    
	suma = approx.sum(axis = 1)
	coord[0] = approx[np.argmin(suma)]
	coord[2] = approx[np.argmax(suma)]

    # Coltul din dreapta sus va avea cea mai mica diferenta iar cel din dreapta jos cea mai mare dif

	diff = np.diff(approx, axis = 1)
	coord[1] = approx[np.argmin(diff)]
	coord[3] = approx[np.argmax(diff)]
	# return the ordered coordinates
	return coord

approx = order_points(approx)
(st_sus, dr_sus, dr_jos, st_jos) = approx #obtinem coordonatele colturilor si apoi calculam distanta si lungimeam maxima


Latime1 = math.dist(dr_jos,st_jos) 
#Latime1 = np.linalg.norm(dr_jos-st_jos) #pentru versiunile mai vechi de python nu exista math.dist asa ca putem folosi numpy
Latime2 = math.dist(dr_sus,st_sus)
#Latime2 = np.linalg.norm(dr_sus-st_sus)
Latime_max = max(int(Latime1), int(Latime2))

Lungime1 = math.dist(dr_sus,dr_jos)
#Lungime1 = np.linalg.norm(dr_sus-dr_jos)
Lungime2 = math.dist(st_sus,st_jos)
#Lungime2 =  np.linalg.norm(st_sus-st_jos)
Lungime_max = max(int(Lungime1), int(Lungime2))

dst = np.array([[0, 0], [Latime_max - 1, 0], [Latime_max - 1, Lungime_max - 1], [0, Lungime_max - 1]],dtype='float32') #cream un vector pentru a obtine o perspectiva de sus a imaginii

M = cv2.getPerspectiveTransform(approx, dst) #ne folosim de coordonatele colturilor si de perspectiva pentru a obtine imaginea finala
warped = cv2.warpPerspective(img_orig, M, (Latime_max, Lungime_max))


pil_image = Image.fromarray(warped)
enhancer = ImageEnhance.Contrast(pil_image)
factor = 1.5 #increase contrast
im_output = enhancer.enhance(factor)


fig = plt.figure()

fig.add_subplot(2,2,1)
plt.imshow(image)
fig.add_subplot(2,2,2)
plt.imshow(edges,cmap = 'gray')
fig.add_subplot(2,2,3)
plt.imshow(contoured_image)
fig.add_subplot(2,2,4)
plt.imshow(im_output)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

class GaussNoisyImg(object):
	'''
	n: number of blobs
	prob: probability of noise
	rad: radius of blobs
	'''
	def __init__(self,n,prob,rad):

		self.n=n
		self.prob=prob
		self.rad=rad

	def blob_image(self):
		img=np.zeros((512,512),dtype='uint8')
		coors=[]
		for i in range(2*self.n):
			coors.append(np.random.randint(10,490))
		ang=np.arange(0,2*np.pi,0.01)
		self.rad=15
		for k in range(0,len(coors),2):
			for i in ang:
				#rad=np.random.randint(20)
				for j in range(self.rad):
					x=int(np.floor(coors[k] + (j * np.cos(i))))
					y=int(np.floor(coors[k+1] + (j * np.sin(i))))
					if(img[x,y] < 240):
						img[x,y] += 40
		#plt.imshow(img,cmap='gray')
		#plt.show()
		return img

	def noise(self):
		image=self.blob_image()
		output=np.zeros(image.shape,np.uint8)
		thres=1-self.prob
		for i in range(image.shape[0]):
			for j in range(image.shape[1]):
				rdn=random.random()
				if(rdn < self.prob):
					output[i][j]=0
				elif(rdn > thres):
					output[i][j]=255
				else:
					output[i][j]=image[i][j]
		#plt.imshow(output,cmap='gray')
		#plt.show()
		blur=cv2.GaussianBlur(output,(5,5),10)
		
		plt.figure(figsize=(20,20))
		plt.subplot(1,3,1)
		plt.title("Orginal Simulated Image")
		plt.imshow(image,cmap='gray')
		plt.subplot(1,3,2)
		plt.title("After adding Gaussian Noise")
		plt.imshow(output,cmap='gray')
		plt.subplot(1,3,3)
		plt.title("After Filtering")
		plt.imshow(blur,cmap='gray')
		plt.show()
		
		return blur
	
	def write_img(self):
		final=self.noise()
		cv2.imwrite('noise_img.png',final)
		return final



obj=GaussNoisyImg(5,0.05,15)
im=obj.write_img()
#plt.imshow(im,cmap='gray')
#plt.show()
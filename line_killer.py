import cv2
import scipy.misc
import numpy as np
from scipy import signal

class Img_preprocess:
    def __init__(self,img=None,filepath=None):
        self.img = img
        self.filepath = filepath
        if(self.img==None):
            self.img = cv2.imread(filepath)
    
    def get_pixel(self,i,j):
        list_pixel = []
        if(i+1<self.img.shape[0]):
            a = self.img[i+1][j]
            if not(a[1]>a[0] or a[1]>a[2]):
                list_pixel.append(a)
            if(j+1<self.img.shape[1]):
                a = self.img[i+1][j+1]
                if not(a[1]>a[0] or a[1]>a[2]):
                    list_pixel.append(a)
        if(j+1<self.img.shape[1]):
            a = self.img[i][j+1]
            if not(a[1]>a[0] or a[1]>a[2]):
                list_pixel.append(a)
            if(i-1>0):
                a = self.img[i-1][j+1]
                if not(a[1]>a[0] or a[1]>a[2]):
                    list_pixel.append(a)
        if(i-1>0):
            a = self.img[i-1][j]
            if not(a[1]>a[0] or a[1]>a[2]):
                list_pixel.append(a)
            if(j-1>0):
                a = self.img[i-1][j-1]
                if not(a[1]>a[0] or a[1]>a[2]):
                    list_pixel.append(a)
        if(j-1>0):
            a = self.img[i][j-1]
            if not(a[1]>a[0] or a[1]>a[2]):
                list_pixel.append(a)
            if(i-1>0):
                a = self.img[i-1][j-1]
                if not(a[1]>a[0] or a[1]>a[2]):
                    list_pixel.append(a)
        return list_pixel

    def find_mean(self,mylist):
        if(len(mylist)==0):
            return 0
        sum = [0,0,0]
        for i in mylist:
            sum[0]+=i[0]
            sum[1]+=i[1]
            sum[2]+=i[2]
        length = len(mylist)
        sum[0]=sum[0]/length
        sum[1]=sum[1]/length
        sum[2]=sum[2]/length
        return sum

    def save_img(self,x,name='outfile.jpg'):
        scipy.misc.imsave(name,x)



    def clear_green(self,region='ha1'):
        if(region =='se1'):
            img_new = np.delete(self.img,200,0)
            img_new = np.delete(img_new,399,0)
            img_new = np.delete(img_new,598,0)
            img_new = np.delete(img_new, [0,200,400,600], axis=1)
            img_new = np.delete(img_new,0,0)
        if(region == 'ha1'):
            img_new = np.delete(self.img,320,0)
            img_new = np.delete(img_new, [25,525,], axis=1)
        self.img=img_new
        for i in enumerate(self.img):
            c_i = i[0]
            for j in enumerate(i[1]):
                c_j = j[0]
                a = j[1]
                p_list = []
                if a[1]>a[0] or a[1]>a[2]:
                    #img[c_i][c_j] = img[c_i][c_j-1] 
                    mean = self.find_mean(self.get_pixel(c_i,c_j))
                    self.img[c_i][c_j]=mean
        self.img = self.img[:-15,:,:]
        return self.img

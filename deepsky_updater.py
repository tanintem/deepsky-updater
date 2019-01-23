#!/usr/bin/env python
# coding: utf-8

# In[13]:


#import scraper 
from scraper import exit_handler,Scraper

from line_killer import Img_preprocess
from shutil import copyfile
import os
import shutil

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
import torch
import numpy as np
import os
import sys
import torch
from torchvision import transforms, datasets
import torch.optim as optim
import matplotlib.pyplot as plt
from CovLstm_cell_simply import ConvLSTMCell as Covlstm_cell
from time import sleep


# # 1 hr model

# In[14]:


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Covlstm_cell(1,1)
        self.decoder = Covlstm_cell(1,1)
        #self.hidden_size = hidden_size
        #self.Convlstm_cell = Covlstm_cell(num_cell,hidden_size)
        #self.relu = nn.ReLU()
    def forward(self,data,epoch,T_en,T_de): #T_en = input sequence, T_de = output sequence
        encoder_state = None
        decoder_state = None
        for t in range(epoch, T_en):
              encoder_state = self.encoder.forward(data[t],encoder_state)
        decoder_input = encoder_state[0][0]
        decoder_input = decoder_input[:,None,:,:] 
        for t in range(0,T_de) :
            decoder_state = self.decoder.forward(decoder_input,decoder_state)
        y_pre = decoder_state[0][0][0]
        # Dont care about hidden states
        return y_pre


# In[15]:


T_de = 20
lr = 0.001
print('Instantiate model')
m = model().cuda()
print(repr(m))

lendata = 100
seq = 20
lenpredict = 6

print('Create a MSE criterion')
loss_fn = nn.MSELoss().cuda()
print(loss_fn)

params = list(m.parameters()) 
print('optimizer Adam')
optimizer = optim.Adam(params, lr=lr)
print(optimizer)
index = 0
index_last_x = 0
m.load_state_dict(torch.load('model/model1.pth'))


# # 3 hr model

# In[16]:


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1_1 = Covlstm_cell(1,1)
        self.rnn1_2 = Covlstm_cell(1,1)
        self.rnn1_3 = Covlstm_cell(1,1)
        self.rnn2_1 = Covlstm_cell(1,1)       
    def init_hiden(self):
        hidden = []
        hidden1_1 = None
        hidden1_2 = None
        hidden1_3 = None        
        hidden2_1 = None
        hidden.append(hidden1_1)
        hidden.append(hidden1_2)
        hidden.append(hidden1_3)
        hidden.append(hidden2_1)
        return hidden       
    def forward(self,data,hidden):
        hidden1_1 = hidden[0]
        hidden1_2 = hidden[1]
        hidden1_3 = hidden[2]        
        hidden2_1 = hidden[3] 
        hidden1_1 = self.rnn1_1.forward(data ,hidden1_1)               
        hidden1_2_input = hidden1_1[0][0]
        hidden1_2_input = hidden1_2_input[:,None,:,:] 
        hidden1_2 = self.rnn1_2.forward(hidden1_2_input,hidden1_2)       
        hidden1_3_input = hidden1_2[0][0]
        hidden1_3_input = hidden1_3_input[:,None,:,:] 
        hidden1_3 = self.rnn1_3.forward(hidden1_3_input,hidden1_3)      
        hidden2_1_input = hidden1_3[0][0]
        hidden2_1_input = hidden2_1_input[:,None,:,:] 
        hidden2_1 = self.rnn2_1.forward(hidden2_1_input ,hidden2_1)
        encoder_out = hidden2_1[0]
        hidden = []
        hidden.append(hidden1_1)
        hidden.append(hidden1_2)
        hidden.append(hidden1_3)
        hidden.append(hidden2_1)
        return encoder_out,hidden
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        num_c = [1, 1, 1]
        h = [1,1,1]
        self.rnn1_1 = Covlstm_cell(1,1)
        self.rnn1_2 = Covlstm_cell(1,1)
        self.rnn1_3 = Covlstm_cell(1,1)
        self.rnn2_1 = Covlstm_cell(1,1)
        
    def forward(self,data,hidden_en):
        hidden1_1 = hidden_en[3]
        hidden1_2 = hidden_en[2]
        hidden1_3 = hidden_en[1]     
        hidden2_1 = hidden_en[0]     
        hidden1_1 = self.rnn1_1.forward(data,hidden1_1)                
        hidden1_2_input = hidden1_1[0][0]
        hidden1_2_input = hidden1_2_input[:,None,:,:] 
        hidden1_2 = self.rnn1_2.forward(hidden1_2_input,hidden1_2)        
        hidden1_3_input = hidden1_2[0][0]
        hidden1_3_input = hidden1_3_input[:,None,:,:] 
        hidden1_3 = self.rnn1_3.forward(hidden1_3_input,hidden1_3)
        hidden2_1_input = hidden1_3[0][0]
        hidden2_1_input = hidden2_1_input[:,None,:,:] 
        hidden2_1 = self.rnn2_1.forward(hidden2_1_input ,hidden2_1)
        out = hidden2_1[0]
        hidden = []
        hidden.append(hidden1_1)
        hidden.append(hidden1_2)
        hidden.append(hidden1_3)
        hidden.append(hidden2_1)
        return out,hidden


# In[17]:


class TraModel(nn.Module):
    def __init__(self):
        super().__init__()
        #input_size_c = 1 hidden_size = h
        self.enc  = Encoder().cuda()
        self.dec  = Decoder().cuda()
        
    def forward(self,data,epoch):
        hidden_en = self.enc.init_hiden()
        T_en = 9 # same seq
        T_en = T_en+epoch
        for t in range(epoch, T_en):
            enc_output,hidden_en = self.enc(data[t],hidden_en)
        #self.dec.init_h0(hidden_en)
        dec_output = enc_output

        for t in range(epoch, T_en):
            dec_output,hidden_en= self.dec(dec_output,hidden_en)
        dec_output = dec_output[0][0]
        return dec_output


# In[18]:


model_3hr = TraModel()
dic_param = torch.load('model/model8_10seq_10000dataset.pt')
model_3hr.load_state_dict(dic_param)


# # load data

# In[19]:


from matplotlib.image import imread
import matplotlib.image as mpimg
from skimage import color
from scipy import ndimage, misc
import cloudy
def load_images(image_paths):
    # Load the images from disk.
    images = [color.rgb2gray(imread(path)) for path in image_paths]
    
    # Convert to a numpy array and return it.
    return np.asarray((images), dtype=np.float32)
current =sorted(cloudy.get_data_dir('input'))
current=current[-20:]


# # Predict&save

# In[20]:



import scipy.misc
def predict_and_save(train_dir):
    train_data = load_images(train_dir)
    Nx_input = torch.from_numpy(train_data).cuda()
    torch.manual_seed(0)
    x_input = Nx_input[:]/255
    x_input = x_input[:,None,None,:,:]
    x_input = Variable(x_input).cuda()

    epoch=0
    T_en = 20
    T_en = T_en+epoch
    output = m(x_input,epoch,T_en,T_de)
    img = output.cpu()
    img = img.data.numpy()
    #img.shape
    img = img*255
    image_name = "prediction/"+train_dir[-1][-16:]
    print(image_name)
    scipy.misc.imsave(image_name, img)
    return img


# In[21]:


def predict_and_save_3hr(train_dir):
    seq = 9
    train_data = load_images(train_dir)
    Nx_input = torch.from_numpy(train_data).cuda()
    torch.manual_seed(0)
#     x_input = Nx_input[:]/255
    x_input = Nx_input[:,None,None,:,:]
    x_input = Variable(x_input).cuda()

    epoch=0
    T_en = 9
    #T_en = T_en+epoch
    output = model_3hr(x_input,epoch)
    img = output.cpu() 
    img = img.data.numpy()
    #img.shape
    img = img*255
    image_name = "prediction_3hr/"+train_dir[-1][-16:] 
    print(image_name)
    scipy.misc.imsave(image_name, img)
    return img


# # ftp & sched

# In[22]:


### ftp
from ftplib import FTP

def placeFile(ftp,filename,filedir):
    try:
        ftp.delete(filename)
    except Exception:
        print("no file to replace")
    status = ftp.storbinary('STOR '+ filename, open(filedir,'rb')) #rb
    print("upload file",filename)
    print(status)

#sched
import sched, time
s = sched.scheduler(time.time, time.sleep)


# In[23]:


def real_time_update(sc):
    ##scrap part
#     try:
    print("try scraping")
    SE1 = Scraper(region = "se1",
                    root_region_url = "http://www.data.jma.go.jp/mscweb/data/himawari/list_se1.html",
                    base_url = "http://www.data.jma.go.jp/mscweb/data/himawari/",
                    verbose = True)

    try:
        img_name=SE1.scraping()
    except KeyboardInterrupt:
        #rint("fail scraping")
        exit_handler(SE1)
    #    
    if img_name is not None:
        path = os.path.abspath('input1/'+img_name)
        print("store image")
        time.sleep(10)

        #store original image
        ftp = FTP('waws-prod-dm1-119.ftp.azurewebsites.windows.net')
        ftp.login(user='deepsky\$deepsky',
                passwd='')
        ftp.cwd('site/public/storage/images')
        placeFile(ftp,filename=img_name,filedir='input1/'+img_name)
        time.sleep(60)
        print("preprocess")
        #preprocess
        preprocess = Img_preprocess(filepath = path)
        try:
            img = preprocess.clear_green(region='se1')
            path = os.path.abspath('input/'+img_name)
            preprocess.save_img(img,name=path)
        except Exception:
            print('preprocess flail',path)
            #
        current.append(path)
        while len(current)>20:
            current.pop(0)
        print("predict  1hr")
        img = predict_and_save(current);
        print("store prediction")
        #store prediction
        ftp.cwd('../')
#         ftp.retrlines('LIST')    
        ftp.cwd('next-1hr')
        placeFile(ftp,filename=img_name,filedir='prediction/'+img_name)        

        print("predict 3 hr")
        img = predict_and_save_3hr(current)
        ftp.cwd('../')
#         ftp.retrlines('LIST')    
        ftp.cwd('next-3hr')
        placeFile(ftp,filename=img_name,filedir='prediction/'+img_name)    
        ftp.quit()
    else:
        print("fail to predict")
#     except Exception:
#         print("fail to scraping")
    s.enter(600, 1, real_time_update, (sc,))


# In[ ]:


s.enter(1, 1, real_time_update, (s,))
s.run()


# In[ ]:





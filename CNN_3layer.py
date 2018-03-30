
# coding: utf-8

# In[1]:


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


import os
from __future__ import print_function
import numpy as np
from vect import get_vect
import pandas as p
import sys


# # Function to extract label from file name

# In[2]:


def lbl_ext(file_name):
    if '\\audio\\' in file_name:
        i = file_name.index('\\audio\\')
        req = file_name[i+7:]
        if '\\' in req:
            w =req.index('\\')
            return(req[:w])


# # Function to extract Features from audio Files

# In[3]:


def feat_ext(file_name):
    features = []
    lbl = []
    file = open(file_name,'r')
    for f in file.readlines():
        e = f.rstrip('\n') 
        name = os.path.abspath(e)
        x = get_vect(name)
        x = x.flatten(order = 'C')

        label = lbl_ext(name)
        features.append(x)
        lbl.append(label)
    return(features,lbl)


# # Function to zero-pad the features to equal size

# In[4]:


def pad(in_features):
    maxval = 0
    for x_val in in_features:
        maxval = max(len(x_val),maxval)
    for i in range (0,len(in_features)):
        if len(in_features[i])<maxval:
            padding_length = maxval-len(in_features[i])
            in_features[i]=np.lib.pad(in_features[i], (0,padding_length), 'constant', constant_values=(0,0))
    in_features = np.array(in_features)
    return(in_features)


# # Convert the label to one hot vector

# In[5]:


def label_onehot(labels):
    POSSIBLE_LABELS = 'yes no up down left right on off stop go Silent'.split()
    req = np.zeros([len(POSSIBLE_LABELS)+1])
    lab_col = []
    lab_dic = {}
    for l in labels:
        if l in POSSIBLE_LABELS:
            ind = POSSIBLE_LABELS.index(l)
            req[ind] = 1
            lab_col.append(req)
            if l not in lab_dic.values():
                lab_dic.update({ind:l})
            req = np.zeros([len(POSSIBLE_LABELS)+1])
        else:
            req[-1] = 1
            lab_col.append(req)
            if 'unknown' not in lab_dic.values():
                lab_dic.update({11:'unknown'})
            req = np.zeros([len(POSSIBLE_LABELS)+1])
    lab_col = np.array(lab_col)
    return(lab_col,lab_dic)


# # Extracting features from Training Files

# In[6]:


train_features,train_label = feat_ext('train_list.txt')


# In[7]:


train_features = pad(train_features)


# In[8]:


train_label,train_dic = label_onehot(train_label)


# # Reshaping the input for CNN

# In[9]:


X = train_features.reshape([-1, 49, 24, 1])
print(X.shape)
Y = train_label
print(Y.shape)


# # Extracting features from Validation files

# In[10]:


val_features,val_label = feat_ext('val_list.txt')
val_features = pad(val_features)
val_label,val_dic = label_onehot(val_label)


# In[11]:


val_X = val_features.reshape([-1,49,24,1])
print(val_X.shape)
val_Y = val_label
print(val_Y.shape)


# # Extracting features from Testing files

# In[12]:


test_features,test_label = feat_ext('test_list.txt')
test_features = pad(test_features)
test_label,test_dic = label_onehot(test_label)
test_X = test_features.reshape([-1,49,24,1])
print(test_X.shape)
test_Y = test_label
print(test_Y.shape)


# # The Convolutional Neural Network Structure
# #### A two convolutional layer network
# #### Each layer is followed by a max pooling layer

# In[15]:


convnet = input_data(shape=[None, 49, 24, 1], name='input')
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 12, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet)
model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_X}, {'targets': test_Y}), 
    snapshot_step=500, show_metric=True, run_id='CNN_Speech')
prediction = model.predict(test_X)


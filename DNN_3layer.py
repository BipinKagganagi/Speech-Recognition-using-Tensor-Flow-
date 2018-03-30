
# coding: utf-8

# In[2]:


import tensorflow as tf
import os
from __future__ import print_function
import numpy as np
from vect import get_vect
import pandas as p


# In[3]:


def lbl_ext(file_name):
    if '\\audio\\' in file_name:
        i = file_name.index('\\audio\\')
        req = file_name[i+7:]
        if '\\' in req:
            w =req.index('\\')
            return(req[:w])


# In[4]:


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


# In[5]:


train_features,train_label = feat_ext('train_list.txt')


# In[6]:


val_features,val_label = feat_ext('val_list.txt')


# In[7]:


test_features,test_label = feat_ext('test_list.txt')


# In[8]:


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


# In[9]:


train_features = pad(train_features)


# In[10]:


val_features = pad(val_features)


# In[11]:


test_features = pad(test_features)


# In[12]:


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


# In[13]:


train_label,train_dic = label_onehot(train_label)


# In[14]:


val_label,val_dic = label_onehot(val_label)


# In[15]:


test_label, test_dic = label_onehot(test_label)


# In[16]:


#hidden layer design
n_nodes_hl1 = 100
n_nodes_hl3 = 100
n_classes = 12


# In[17]:


x = tf.placeholder('float', [None,1176])
y = tf.placeholder('float')


# In[19]:


def neural_network_model(data):
    data = tf.cast(data,tf.float32)
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([1176, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)


    l3 = tf.add(tf.matmul(l1,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return(output)

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 50
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(100):
                i = 0
                j = 510
                epoch_x, epoch_y = train_features[i:j:1], train_label[i:j:1]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                i +=510
                j +=510

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_features, y:test_label}))
train_neural_network(x)


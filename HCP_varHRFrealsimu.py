# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:08:45 2018

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:38:38 2018

@author: Admin
"""
from keras.layers import*
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as T
import sys
import scipy.io as si
import numpy
import matplotlib.pyplot as plt
from keras.constraints import non_neg
import time
nnode = 4
numpy.random.seed(0)
sys.path.insert(0,'C:/Users/Admin/ZY/Python/DeepCCA')
sys.path.insert(0,'C:/Users/Admin/ZY/Python/CommonFun')
def lstmCCA_model(shape_fmri):
    input_fMRI = Input(shape = shape_fmri)
    input_fMRI_dwt = Input(shape = shape_fmri)
    shared_conv1D = Conv1D(nnode,15,padding='same')
    shared_lstm = LSTM(nnode,return_sequences = True)
    shared_lstm_2 = LSTM(1,return_sequences = True)
    shared_tdense = TimeDistributed(Dense(1))
    
    conv1D_fMRI = shared_conv1D(input_fMRI)
    conv1D_fMRI_dwt = shared_conv1D(input_fMRI_dwt)
    lstm_fMRI = shared_lstm(conv1D_fMRI)
    lstm_fMRI_dwt = shared_lstm(conv1D_fMRI_dwt)
    
    tdense_fMRI = shared_tdense(lstm_fMRI)
    tdense_fMRI_dwt = shared_tdense(lstm_fMRI_dwt)
    
#    tdense_fMRI = Flatten()(tdense_fMRI)
#    tdense_fMRI_dwt = Flatten()(tdense_fMRI_dwt)
    
    merged_data = concatenate([lstm_fMRI,lstm_fMRI_dwt],axis = -1)
    model = Model(inputs = [input_fMRI,input_fMRI_dwt],
                  outputs = merged_data)
    return model

def lstmCCA_model_test(shape_fmri):
    input_fMRI = Input(shape = shape_fmri)
    input_fMRI_dwt = Input(shape = shape_fmri)
    shared_conv1D = Conv1D(nnode,15,padding='same')
    shared_lstm = LSTM(nnode,return_sequences = True)
    shared_lstm_2 = LSTM(1,return_sequences = True)
    shared_tdense = TimeDistributed(Dense(1))
    
    conv1D_fMRI = shared_conv1D(input_fMRI)
    conv1D_fMRI_dwt = shared_conv1D(input_fMRI_dwt)
    lstm_fMRI = shared_lstm(conv1D_fMRI)
    lstm_fMRI_dwt = shared_lstm(conv1D_fMRI_dwt)
    
    tdense_fMRI = shared_tdense(lstm_fMRI)
    tdense_fMRI_dwt = shared_tdense(lstm_fMRI_dwt)
    
#    tdense_fMRI = Flatten()(tdense_fMRI)
#    tdense_fMRI_dwt = Flatten()(tdense_fMRI_dwt)
    
    merged_data = concatenate([lstm_fMRI,lstm_fMRI_dwt],axis = -1)
    model = Model(inputs = [input_fMRI,input_fMRI_dwt],
                  outputs = merged_data)
    return model
def lstmCCA_model_nonneg(shape_fmri):
    input_fMRI = Input(shape = shape_fmri)
    input_fMRI_dwt = Input(shape = shape_fmri)
    shared_conv1D = Conv1D(4,5,padding='same',kernel_constraint=non_neg())
    shared_lstm = LSTM(4,return_sequences = True)
    shared_lstm_2 = LSTM(1,return_sequences = True)
    shared_tdense = TimeDistributed(Dense(1))
    
    conv1D_fMRI = shared_conv1D(input_fMRI)
    conv1D_fMRI_dwt = shared_conv1D(input_fMRI_dwt)
    lstm_fMRI = shared_lstm(conv1D_fMRI)
    lstm_fMRI_dwt = shared_lstm(conv1D_fMRI_dwt)
    
    tdense_fMRI = shared_tdense(lstm_fMRI)
    tdense_fMRI_dwt = shared_tdense(lstm_fMRI_dwt)
    
#    tdense_fMRI = Flatten()(tdense_fMRI)
#    tdense_fMRI_dwt = Flatten()(tdense_fMRI_dwt)
    
    merged_data = concatenate([tdense_fMRI,tdense_fMRI_dwt],axis = -1)
    model = Model(inputs = [input_fMRI,input_fMRI_dwt],
                  outputs = merged_data)
    return model
def conv1D_model(shape_fmri):
    input_fMRI = Input(shape = shape_fmri)
    input_fMRI_dwt = Input(shape = shape_fmri)
    shared_conv1D = Conv1D(8,4,padding='same')
    shared_conv1D_2 = Conv1D(1,4,padding='same')
    shared_lstm = Bidirectional(LSTM(10,return_sequences = True))
    shared_tdense = TimeDistributed(Dense(1))
    
    conv1D_fMRI = shared_conv1D(input_fMRI)
    conv1D_fMRI_dwt = shared_conv1D(input_fMRI_dwt)

    
    tdense_fMRI = shared_conv1D_2(conv1D_fMRI)
    tdense_fMRI_dwt = shared_conv1D_2(conv1D_fMRI_dwt)
    
#    tdense_fMRI = Flatten()(tdense_fMRI)
#    tdense_fMRI_dwt = Flatten()(tdense_fMRI_dwt)
    
    merged_data = concatenate([tdense_fMRI,tdense_fMRI_dwt],
                              axis = -1)
    model = Model(inputs = [input_fMRI,input_fMRI_dwt],
                  outputs = merged_data)
    return model
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = T.mean(x,axis=-1,keepdims=True)
    my = T.mean(y,axis=-1,keepdims=True)
    xm, ym = x-mx, y-my
    r_num = T.sum(xm*ym,axis=-1)
    r_den = T.sqrt(T.sum(T.square(xm),axis=-1)* T.sum(T.square(ym),axis = -1))
    r = T.max(r_num / r_den, axis=-1)
    r = T.sum(r)
    return r
def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = numpy.mean(x,axis=-1,keepdims=True)
    my = numpy.mean(y,axis=-1,keepdims=True)
    xm, ym = x-mx, y-my
    r_num = numpy.sum(xm*ym,axis=-1)
    r_den = numpy.sqrt(numpy.sum(numpy.square(xm),axis=-1)* numpy.sum(numpy.square(ym),
                                 axis = -1))
    r = numpy.max(r_num / r_den,axis=-1)
    r_ind = numpy.argmax(r_num / r_den, axis=-1)
    return r, r_ind

def lstm_corr(y_pred,X,pinvX):
    Y = y_pred[:,:,:4] # fMRI
    Y_dwt = y_pred[:,:,4:] # fMRI_dwt
    Y = Y - numpy.mean(Y,axis = 1,keepdims=True)
    Y_dwt = Y_dwt - numpy.mean(Y_dwt,axis = 1,keepdims=True)
    beta = numpy.dot(numpy.transpose(Y,(0,2,1)),pinvX)
    beta_dwt = numpy.dot(numpy.transpose(Y_dwt,(0,2,1)),pinvX)
    Yest = numpy.dot(beta,X.T)
    Yest_dwt = numpy.dot(beta_dwt,X.T)    
    

    corr_Y,r_ind = correlation_coefficient(numpy.transpose(Y,(0,2,1)),Yest)
    corr_Y_dwt,r_dwt_ind = correlation_coefficient(numpy.transpose(Y_dwt,(0,2,1)),Yest_dwt)
    Y = Y[numpy.arange(Y.shape[0]),:,r_ind]
    Y_dwt = Y_dwt[numpy.arange(Y_dwt.shape[0]),:,r_dwt_ind]
    return corr_Y,corr_Y_dwt, Yest, Yest_dwt,beta,beta_dwt,Y,Y_dwt


def lstm_loss(X,pinvX):
    def inner_lstm_loss(y_true,y_pred):
        Y = y_pred[:,:,:4] # fMRI
        Y_dwt = y_pred[:,:,4:] # fMRI_dwt
        Y = Y - T.mean(Y,axis = 1,keepdims=True)
        Y_dwt = Y_dwt - T.mean(Y_dwt,axis = 1,keepdims=True)
        beta = T.dot(T.permute_dimensions(Y,(0,2,1)),pinvX)
        beta_dwt = T.dot(T.permute_dimensions(Y_dwt,(0,2,1)),pinvX)
        Yest = T.dot(beta,X.T)
        Yest_dwt = T.dot(beta_dwt,X.T)
        corr_Y = correlation_coefficient_loss(T.permute_dimensions(Y,(0,2,1)),Yest)
#        return -corr_Y
        corr_Y_dwt = correlation_coefficient_loss(T.permute_dimensions(Y_dwt,(0,2,1)),
                                                  Yest_dwt)
        return corr_Y_dwt-corr_Y
    return inner_lstm_loss

from scipy.stats.mstats import zscore
import readMat
from scipy.io import savemat
datadir = "J:/HCP_simulation/simudata"
from os import listdir, remove
from os.path import isfile, join
subjlist = [f for f in listdir(datadir) if isfile(join(datadir,f))==False]
epochs = 20 # 20 epochs is specified for usual subjects.
for subid,sub in enumerate(subjlist):
    tempdatadir = datadir+"/"+sub+"/realsimu_varHRFsimudatav2.mat" 
    modeldir = datadir+"/"+sub+"/varHRFsimudata_modelv2.h5"
    savedatadir = datadir+"/"+sub+"/varHRFrealsimu_lstmv2.mat"     
    if isfile(savedatadir):
        continue
    fMRIdata_q,fMRIdata_dwt_q,X,mask,Cor = readMat.readMatVars(
            tempdatadir,varname=("fMRIdata","dwt_fMRIdata","X","mask","Cor"))  
    start_time = time.time()
    Cor = numpy.reshape(Cor,(Cor.size,))
    mask_q = numpy.ones((fMRIdata_q.shape[0],))
    perct = numpy.percentile(Cor,90.0)
    mask_q[numpy.logical_and(Cor<perct,Cor>0)==True] = 2
    X = zscore(X.T) # zscore is not used in previous analysis 09/11/2018
    pinvX = numpy.linalg.pinv(X)
    pinvX = pinvX.T
    

    fMRIdata_q = zscore(fMRIdata_q,axis=-1)
    fMRIdata_dwt_q = zscore(fMRIdata_dwt_q,axis=-1);
    fMRIdata_q = numpy.reshape(fMRIdata_q,fMRIdata_q.shape+(1,))
    fMRIdata_dwt_q = numpy.reshape(fMRIdata_dwt_q,fMRIdata_dwt_q.shape+(1,))
    n_q = numpy.sum(mask_q==1)
    
    model = lstmCCA_model(fMRIdata_q.shape[1:])
    opt = Adam(lr=0.01,beta_1=0.9, beta_2 = 0.999, decay = 0.05)
    model.compile(optimizer=opt,loss=lstm_loss(X,pinvX))
    

     
    randval = numpy.random.rand(n_q)
    split = 0.9;

    error_log = numpy.zeros((epochs,4))    
    fMRIdata_q_train = fMRIdata_q[mask_q==1,:,:]
    tempind = numpy.random.permutation(fMRIdata_dwt_q.shape[0])
    fMRIdata_dwt_q_train = fMRIdata_dwt_q[tempind[:n_q],:,:]  
    for e in range(epochs):

        history = model.fit([fMRIdata_q_train[randval<=split,:,:],fMRIdata_dwt_q_train[randval<=split,:,:]],
                            y=numpy.ones((numpy.sum(randval<=split),283,2)),batch_size = 500,epochs = 1)    
        # testing dataset
        lstm_data = model.predict([fMRIdata_q_train,fMRIdata_dwt_q_train],batch_size=500)
        corr_Y,corr_Y_dwt, Yest, Yest_dwt,beta,beta_dwt,Y,Y_dwt = lstm_corr(lstm_data,X,pinvX)            
        error_log[e,0] = numpy.mean(corr_Y[randval<=split])
        error_log[e,1] = numpy.mean(corr_Y_dwt[randval<=split])
        error_log[e,2] = numpy.mean(corr_Y[randval>split])
        error_log[e,3] = numpy.mean(corr_Y_dwt[randval>split])

        print(error_log[e,:])
        
    
    model.save(modeldir)   
    
    lstm_data = model.predict([fMRIdata_q,fMRIdata_q],batch_size=500)
    lstm_dwtdata = model.predict([fMRIdata_dwt_q,fMRIdata_dwt_q],batch_size=500)
    corr_Y,corr_Y_nan, Yest, Yest_dwt,beta,beta_dwt,Y,Y_nan = lstm_corr(lstm_data,X,pinvX)  
    corr_Y_dwt,corr_Y_nan, Yest, Yest_dwt,beta,beta_dwt,Y_dwt,Y_nan = lstm_corr(lstm_dwtdata,X,pinvX)  
    time_cost = time.time()-start_time
    
    savemat(savedatadir,{'fMRIdata':Y,'fMRIdata_dwt':Y_dwt,
                         'corr':corr_Y,'error_log':error_log,'corr_dwt_xyz':corr_Y_dwt,
                         'epochs':epochs,'time':time_cost})

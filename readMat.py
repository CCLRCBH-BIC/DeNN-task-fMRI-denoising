# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:43:22 2017

@author: ZYang
"""

import h5py
import numpy

#import theano.tensor as T
def readMatTrainValidTest(filepath,shared=1):
    A = h5py.File(filepath,'r')
    A_var_list=("x_train","y_train","x_valid","y_valid","x_test","y_test",
                "labels_train","labels_valid","labels_test","x_xyz_train","x_xyz_valid","x_xyz_test")
    x_train = A[A_var_list[0]][()]
    y_train= A[A_var_list[1]][()]
    x_valid= A[A_var_list[2]][()]
    y_valid= A[A_var_list[3]][()]
    x_test= A[A_var_list[4]][()]
    y_test= A[A_var_list[5]][()]
    if shared==1:
        import theano
        # x is 2-D array and y is 1-D array
        x_train = theano.shared(numpy.asarray(x_train,dtype=theano.config.floatX),borrow=True)
        x_valid = theano.shared(numpy.asarray(x_valid,dtype=theano.config.floatX),borrow=True)
        x_test = theano.shared(numpy.asarray(x_test,dtype=theano.config.floatX),borrow=True)
        y_train = theano.shared(numpy.asarray(y_train,dtype='int32'),borrow=True)
        y_valid = theano.shared(numpy.asarray(y_valid,dtype='int32'),borrow=True)
        y_test = theano.shared(numpy.asarray(y_test,dtype='int32'),borrow=True)  
        y_train=y_train.flatten()
        y_valid=y_valid.flatten()
        y_test=y_test.flatten()
    elif shared==2:
        # x is 2-D array and y is multilabel-D array
        y_train=A[A_var_list[6]][()]
        y_valid=A[A_var_list[7]][()]
        y_test=A[A_var_list[8]][()]
    elif shared==3:
        # x is multi-D array and y is multilabel-D array
        x_train=A[A_var_list[9]][()]
        x_valid=A[A_var_list[10]][()]
        x_test=A[A_var_list[11]][()]
        y_train=A[A_var_list[6]][()]
        y_valid=A[A_var_list[7]][()]
        y_test=A[A_var_list[8]][()]
    return (x_train,y_train), (x_valid,y_valid),(x_test,y_test)
        
def readMatVars(filepath,varname):
    """
    varname: a tuple of variables to load
    return:
        a list of ndarray
    """
    A = h5py.File(filepath,'r')
    var = list()
    for i in range(len(varname)):
        temp = A[varname[i]].value #[()]
        var.append(temp)
    return var

def splitgroup(Nsub, ratio = [5,1,1]):
    group_ind = numpy.zeros((Nsub,))
    group_id = range(Nsub)#numpy.random.permutation(range(Nsub))
    start_ind = 0
    
    for i in range(len(ratio)):
        end_ind = int(start_ind + 1.*Nsub/sum(ratio)*ratio[i])
        end_ind = numpy.min([Nsub, end_ind])
        group_ind[group_id[start_ind:end_ind]] = i + 1
        start_ind = end_ind + 1
    return group_ind

def KFold(Nsub, nfold,rand = 1):
    """
    fold id vary from 0 to nfold-1
    """
    fold_id = numpy.zeros((Nsub,))
    if rand==0:
        per_ind = range(Nsub)
    else:
        per_ind = numpy.random.permutation(Nsub)
    subperfold = Nsub //nfold
    for i in range(nfold):
        fold_id[per_ind[i*subperfold:numpy.min([(i+1)*subperfold,Nsub])]] = i
    return fold_id
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:21:45 2019

@author: rfablet
"""

import keras
import numpy as np
#from keras import backend as K
#import tensorflow as tf
#import matplotlib.pyplot as plt 
#import os
import argparse
#import pickle
from sklearn import decomposition
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import os
    
from keras.constraints import Constraint
from keras import backend as K
       
 
#os.chdir('../Utils')
#import OI        

#import dill as pickle



# main code
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #parser.add_argument('-d', '--data', help='Image dataset used for learning: cifar, mnist or custom numpy 4D array (samples, rows, colums, channel) using pickle', type=str, default='cifar')
    #parser.add_argument('-e', '--epoch', help='Number of epochs', type=int, default=100)
    #parser.add_argument('-b', '--batchsize', help='Batch size', type=int, default=64)
    #parser.add_argument('-o', '--output', help='Output model name (required) (.h5)', type=str, required = True)
    #parser.add_argument('--optim', help='Optimization method: sgd, adam', type=str, default='sgd')

    flagDisplay   = 0
    flagProcess   = [0,1,2,3,4]
    flagSaveModel = 1
    flagTrOuputWOMissingData = 1
    flagloadOIData = 0
    
    flagDataset = 2
    Wsquare     = 4#0 # half-width of holes
    Nsquare     = 3  # number of holes
    DimAE       = 40#20 # Dimension of the latent space
    flagAEType  = 7#7#
    flagOptimMethod = 0 # 0 DINAE : iterated projections, 1 : Gradient descent  
    flagGradModel   = 0 # 0: F(Grad,Mask), 1: F==(Grad,Grad(t-1),Mask), 2: LSTM(Grad,Mask)
    sigNoise        = 1e-1
    
    flagUseMaskinEncoder = 0
    stdMask              = 0.
    alpha                = np.array([1.,0.,0.])
    
    flagDataWindowing = 0 # 2 for SST case-study

    dropout           = 0.0
    wl2               = 0.0000
    batch_size        = 4#4#8#12#8#256#8
    NbEpoc            = 20
    Niter             = 50
    NSampleTr         = 445#550#334#
        
    flag_MultiScaleAEModel = 0 # see AE Type 007
    
    # functions for the evaluation of interpolation and auto-encoding performance
    def eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt):
    
        mse_AE_Tr        = np.mean( (rec_AE_Tr - x_train)**2 )
        var_Tr           = np.mean( (x_train-np.mean(x_train,axis=0)) ** 2 )
        exp_var_AE_Tr    = 1. - mse_AE_Tr / var_Tr
        
        mse_AE_Tt        = np.mean( (rec_AE_Tt - x_test)**2 )
        var_Tt           = np.mean( (x_test-np.mean(x_train,axis=0))** 2 )
        exp_var_AE_Tt    = 1. - mse_AE_Tt / var_Tt
                
        return exp_var_AE_Tr,exp_var_AE_Tt
    
    def eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,
                               mask_test,x_test,x_test_missing,x_test_pred):
        mse_train      = np.zeros((2))
        mse_train[0]   = np.sum( mask_train * (x_train_pred - x_train_missing)**2 ) / np.sum( mask_train )
        mse_train[1]   = np.mean( (x_train_pred - x_train)**2 )
        #var_Tr        = np.sum( mask_train * (x_train_missing-np.mean(np.mean(x_train_missing,axis=0),axis=0)) ** 2 ) / np.sum( mask_train )
        exp_var_train  = 1. - mse_train #/ var_Tr
                
        mse_test        = np.zeros((2))
        mse_test[0]     = np.sum( mask_test * (x_test_pred - x_test_missing)**2 ) / np.sum( mask_test )
        mse_test[1]     = np.mean( (x_test_pred - x_test)**2 ) 
        #var_Tt       = np.sum( mask_test * (x_test_missing-np.mean(np.mean(x_train_missing,axis=0),axis=0))** 2 ) / np.sum( mask_test )
        exp_var_test = 1. - mse_test #/ var_Tt

        mse_train_interp        = np.sum( (1.-mask_train) * (x_train_pred - x_train)**2 ) / np.sum( 1. - mask_train )
        exp_var_train_interp    = 1. - mse_train_interp #/ var_Tr
        
        mse_test_interp        = np.sum( (1.-mask_test) * (x_test_pred - x_test)**2 ) / np.sum( 1. - mask_test )
        exp_var_test_interp    = 1. - mse_test_interp #/ var_Tr
                
        return mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp
    
    def defineClassifier(DimAE,num_classes):
        ## Learning a classifier
        
        classifier = keras.Sequential()
        classifier.add(keras.layers.Dense(32,activation='relu', input_shape=(DimAE,)))
        classifier.add(keras.layers.Dense(64,activation='relu'))
        classifier.add(keras.layers.Dense(num_classes, activation='softmax'))
        
        return classifier
    
    def define_DINConvAE(NiterProjection,model_AE,shape,alpha,flagDisplay=0):
    
      # encoder-decoder with masked data
      x_input         = keras.layers.Input((shape[1],shape[2],shape[3]))
      mask            = keras.layers.Input((shape[1],shape[2],shape[3]))
        
      x     = keras.layers.Lambda(lambda x:1.*x)(x_input)
      mask_ = keras.layers.Lambda(lambda x:1.-x)(mask)
    
      # Iterations of fixed-point projection
      for kk in range(0,NiterProjection):
          x_proj   = model_AE([x,mask])
          x_proj   = keras.layers.Multiply()([x_proj,mask_])
          x        = keras.layers.Multiply()([x,mask])
          x        = keras.layers.Add()([x,x_proj])
    
      if flag_MultiScaleAEModel == 1:
          x_proj,x_projLR = model_AE_MR([x,mask])
          global_model_FP    = keras.models.Model([x_input,mask],[x_proj])
          global_model_FP_MR = keras.models.Model([x_input,mask],[x_proj,x_projLR])
      else:
          x_proj = model_AE([x,mask])
          global_model_FP    = keras.models.Model([x_input,mask],[x_proj])

    
      x_input         = keras.layers.Input((shape[1],shape[2],shape[3]))
      mask            = keras.layers.Input((shape[1],shape[2],shape[3]))
    
      # randomly sample an additionnal missing data mask
      # additive noise + spatial smoothing
      if flagUseMaskinEncoder == 1:
          WAvFilter     = 3
          NIterAvFilter = 3
          thrNoise      = 1.5 * stdMask + 1e-7
          maskg   = keras.layers.GaussianNoise(stdMask)(mask)
          
          avFilter       = 1./(WAvFilter**3)*np.ones((WAvFilter,WAvFilter,WAvFilter,1,1))
          spatialAvLayer = keras.layers.Conv3D(1,(WAvFilter,WAvFilter,WAvFilter),weights=[avFilter],padding='same',activation='linear',use_bias=False,name='SpatialAverage')
          spatialAvLayer.trainable = False
          maskg = keras.layers.Lambda(lambda x: K.permute_dimensions(x,(0,3,1,2)))(maskg) 
     
          maskg  = keras.layers.Reshape((shape[3],shape[1],shape[2],1))(maskg)
          for nn in range(0,NIterAvFilter):
              maskg  = spatialAvLayer(maskg) 
          maskg = keras.layers.Lambda(lambda x: K.permute_dimensions(x,(0,2,3,1,4)))(maskg) 
          maskg = keras.layers.Reshape((shape[1],shape[2],shape[3]))(maskg)
              
          def thresholding(x,thr):    
            greater = K.greater_equal(x,thr) #will return boolean values
            greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1    
            return greater 
          maskg = keras.layers.Lambda(lambda x: thresholding(x,thrNoise))(maskg)    
         
          maskg  = keras.layers.Multiply()([mask,maskg])
          maskg  = keras.layers.Subtract()([mask,maskg])       
      else:
          maskg = keras.layers.Lambda(lambda x: 1.*x)(mask)
          
      if flag_MultiScaleAEModel == 0:
          x_proj = global_model_FP([x_input,maskg])
      else:
          x_proj,x_projLR = global_model_FP_MR([x_input,maskg])
      #x_proj = keras.layers.Multiply()([x_proj,mask])
      
      # AE error with x_proj
      err1   = keras.layers.Subtract()([x_proj,x_input])
      err1   = keras.layers.Multiply()([err1,mask])
      err1   = keras.layers.Multiply()([err1,err1])
      err1   = keras.layers.Reshape((shape[1],shape[2],shape[3],1))(err1)
      err1   = keras.layers.GlobalAveragePooling3D()(err1)
      err1   = keras.layers.Reshape((1,))(err1)
      err1   = keras.layers.Lambda(lambda x:alpha[0]*x)(err1)
      
      # AE error with x_proj
      if flag_MultiScaleAEModel == 1:
          err1LR   = keras.layers.Subtract()([x_projLR,x_input])
          err1LR   = keras.layers.Multiply()([err1LR,mask])
          err1LR   = keras.layers.Multiply()([err1LR,err1LR])
          err1LR   = keras.layers.Reshape((shape[1],shape[2],shape[3],1))(err1LR)
          err1LR   = keras.layers.GlobalAveragePooling3D()(err1LR)
          err1LR   = keras.layers.Reshape((1,))(err1LR)
          err1LR   = keras.layers.Lambda(lambda x:alpha[0]*x)(err1LR)
          err1     = keras.layers.Add()([err1,err1LR])

      # compute error (x_proj-x_input)**2 with full-1 mask
      x_proj2 = model_AE([x_proj,keras.layers.Lambda(lambda x:1.-0.*x)(mask)])
      err2    = keras.layers.Subtract()([x_proj,x_proj2])
      err2    = keras.layers.Multiply()([err2,err2])
      err2   = keras.layers.Reshape((shape[1],shape[2],shape[3],1))(err2)
      err2   = keras.layers.GlobalAveragePooling3D()(err2)
      err2   = keras.layers.Reshape((1,))(err2)
      err2   = keras.layers.Lambda(lambda x:alpha[1]*x)(err2)
    
      # compute error (x_proj-x_input)**2 with full-1 mask
      x_proj3 = model_AE([x_proj,keras.layers.Lambda(lambda x:0.*x)(mask)])
      err3    = keras.layers.Subtract()([x_proj3,x_proj])
      err3    = keras.layers.Multiply()([err3,err3])
      err3    = keras.layers.Reshape((shape[1],shape[2],shape[3],1))(err3)
      err3    = keras.layers.GlobalAveragePooling3D()(err3)
      err3    = keras.layers.Reshape((1,))(err3)
      err3    = keras.layers.Lambda(lambda x:alpha[2]*x)(err3)

      err    = keras.layers.Add()([err1,err2])
      err    = keras.layers.Add()([err,err3])
      global_model_FP_Masked  = keras.models.Model([x_input,mask],err)
    
      if flagDisplay == 1:
          global_model_FP.summary()
          global_model_FP_Masked.summary()
      
      return global_model_FP,global_model_FP_Masked
    
    def define_GradModel(model_AE,shape,flagGradModel=0,flagDisplay=1):
      x_input         = keras.layers.Input((shape[1],shape[2],2*shape[3]))
      
      x = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x_input)
      for nn in range(0,10):
          x = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)

#      x        = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x_input)
#      x        = keras.layers.Conv2D(5,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
#      for nn in range(0,6):
#          dx = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
#          dx = keras.layers.Conv2D(5,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
#          x  = keras.layers.Add()([x,dx])
          
      x = keras.layers.Conv2D(shape[3],(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
      #x = keras.layers.Reshape((shape[1],shape[2]))(x)
      gradMaskModel  = keras.models.Model([x_input],[x])
        
      # conv model for gradient      
      if flagGradModel == 0:
          print("...... Initialize Gradient Model: ResNet")
          x_input  = keras.layers.Input((shape[1],shape[2],2*shape[3]))
          x        = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x_input)
          x        = keras.layers.Conv2D(5,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
          for nn in range(0,10):
              dx = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              dx = keras.layers.Conv2D(5,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
              x  = keras.layers.Add()([x,dx])
          x = keras.layers.Conv2D(shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
          #x = keras.layers.Reshape((shape[1],shape[2]))(x)
          gradModel  = keras.models.Model([x_input],[x])

      elif flagGradModel == 1:
          print("...... Initialize Gradient Model: ResNet with one-step memory")
          x_input  = keras.layers.Input((shape[1],shape[2],shape[3]*2))
          x        = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x_input)
          x        = keras.layers.Conv2D(5,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
          for nn in range(0,10):
              dx = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              dx = keras.layers.Conv2D(5,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
              x  = keras.layers.Add()([x,dx])
          x = keras.layers.Conv2D(shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
          #x = keras.layers.Reshape((shape[1],shape[2]))(x)
          gradModel  = keras.models.Model([x_input],[x])
          
      elif flagGradModel == 2:
          print("...... Initialize Gradient Model: LSTM")
          #layer_LSTM = keras.layers.ConvLSTM2D(10, kernel_size=(1,3), padding='same', use_bias=False, activation='relu')#,return_state=True)#,input_shape=(1, para_Model.NbHS))          
          x_input  = keras.layers.Input((shape[1],shape[2],shape[3]*2))

          gx_lstm  = keras.layers.ConvLSTM2D(10, kernel_size=(1,3), padding='same', use_bias=False, activation='relu')(keras.layers.Reshape((1,shape[1],shape[2],2))(x_input))#,return_state=True)#,input_shape=(1, para_Model.NbHS))
          gx_lstm  = keras.layers.Conv2D(shape[3],(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(gx_lstm)
          
          #gx_lstm = keras.layers.Reshape((shape[1],shape[2]))(gx_lstm)
          gradModel  = keras.models.Model([x_input],[gx_lstm])
 
      if flagDisplay == 1:
          gradMaskModel.summary()
          gradModel.summary()
      return gradModel,gradMaskModel
  
    def define_GradDINConvAE(NiterProjection,NiterGrad,model_AE,shape,gradModel,gradMaskModel,flagGradModel=0,flagDisplay=0):

      # encoder-decoder with masked data
      x_input         = keras.layers.Input((shape[1],shape[2],shape[3]))
      mask            = keras.layers.Input((shape[1],shape[2],shape[3]))
    
      x     = keras.layers.Lambda(lambda x:1.*x)(x_input)
      mask_ = keras.layers.Lambda(lambda x:1.-x)(mask)
    
      # fixed-point projections as initialization
      for kk in range(0,NiterProjection):
          x_proj   = model_AE([x,mask])
          x_proj   = keras.layers.Multiply()([x_proj,mask_])
          x        = keras.layers.Multiply()([x,mask])
          x        = keras.layers.Add()([x,x_proj])
    
      # gradient descent
      for kk in range(0,NiterGrad):
          x_proj   = model_AE([x,mask])
          dx       = keras.layers.Subtract()([x,x_proj])
          
          # grad mask
          dmask    = keras.layers.Concatenate(axis=-1)([dx,mask_])
          dmask    = gradMaskModel(dmask)
          
          # grad update
          #gx    = keras.layers.Concatenate(axis=-1)([keras.layers.Reshape((shape[1],shape[2],1))(dx),keras.layers.Reshape((shape[1],shape[2],1))(mask_)])
          #gx    = gradModel(gx)
          if flagGradModel == 0:
              gx    = keras.layers.Concatenate(axis=-1)([dx,mask_])
              gx    = gradModel(gx)
          elif flagGradModel == 1:
              if kk == 0:
                  gx = keras.layers.Lambda(lambda x:0.*x)(dx)              
              gx    = keras.layers.Concatenate(axis=-1)([mask_,gx])                  
              gx    = keras.layers.Concatenate(axis=-1)([dx,gx])

              gx    = gradModel(gx)
          elif flagGradModel == 2:
              gx    = keras.layers.Concatenate(axis=-1)([dx,mask_])
              gx    = gradModel(gx)
              
          dx    = keras.layers.Multiply()([gx,dmask])
          xnew  = keras.layers.Add()([x,dx])
          xnew  = keras.layers.Multiply()([xnew,mask_])
          
          # update with masking
          x        = keras.layers.Multiply()([x,mask])
          x        = keras.layers.Add()([x,xnew])
                  
      x_proj = model_AE([x,mask])
      global_model_Grad  = keras.models.Model([x_input,mask],[x_proj])
    
      x_input         = keras.layers.Input((shape[1],shape[2],shape[3]))
      mask            = keras.layers.Input((shape[1],shape[2],shape[3]))
    
      # randomly sample an additionnal missing data mask
      # additive noise + spatial smoothing
      if flagUseMaskinEncoder == 1:
          WAvFilter     = 3
          NIterAvFilter = 3
          thrNoise      = 1.5 * stdMask + 1e-7
          maskg   = keras.layers.GaussianNoise(stdMask)(mask)
          
          avFilter       = 1./(WAvFilter**3)*np.ones((WAvFilter,WAvFilter,WAvFilter,1,1))
          spatialAvLayer = keras.layers.Conv3D(1,(WAvFilter,WAvFilter,WAvFilter),weights=[avFilter],padding='same',activation='linear',use_bias=False,name='SpatialAverage')
          spatialAvLayer.trainable = False
          maskg = keras.layers.Lambda(lambda x: K.permute_dimensions(x,(0,3,1,2)))(maskg) 
     
          maskg  = keras.layers.Reshape((shape[3],shape[1],shape[2],1))(maskg)
          for nn in range(0,NIterAvFilter):
              maskg  = spatialAvLayer(maskg) 
          maskg = keras.layers.Lambda(lambda x: K.permute_dimensions(x,(0,2,3,1,4)))(maskg) 
          maskg = keras.layers.Reshape((shape[1],shape[2],shape[3]))(maskg)
              
          def thresholding(x,thr):    
            greater = K.greater_equal(x,thr) #will return boolean values
            greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1    
            return greater 
          maskg = keras.layers.Lambda(lambda x: thresholding(x,thrNoise))(maskg)    
         
          maskg  = keras.layers.Multiply()([mask,maskg])
          maskg  = keras.layers.Subtract()([mask,maskg])       
      else:
          maskg = keras.layers.Lambda(lambda x: 1.*x)(mask)

      x_proj = global_model_Grad([x_input,maskg])
      #x_proj = keras.layers.Multiply()([x_proj,mask])
      #x_proj = global_model_FP([x_input,maskg])
      #x_proj = keras.layers.Multiply()([x_proj,mask])
      
      # AE error with x_proj
      err1   = keras.layers.Subtract()([x_proj,x_input])
      err1   = keras.layers.Multiply()([err1,mask])
      err1   = keras.layers.Multiply()([err1,err1])
      err1   = keras.layers.Reshape((shape[1],shape[2],shape[3],1))(err1)
      err1   = keras.layers.GlobalAveragePooling3D()(err1)
      err1   = keras.layers.Reshape((1,))(err1)
      err1   = keras.layers.Lambda(lambda x:alpha[0]*x)(err1)
      
      # compute error (x_proj-x_input)**2 with full-1 mask
      x_proj2 = model_AE([x_proj,keras.layers.Lambda(lambda x:1.-0.*x)(mask)])
      err2    = keras.layers.Subtract()([x_proj,x_proj2])
      err2    = keras.layers.Multiply()([err2,err2])
      err2   = keras.layers.Reshape((shape[1],shape[2],shape[3],1))(err2)
      err2   = keras.layers.GlobalAveragePooling3D()(err2)
      err2   = keras.layers.Reshape((1,))(err2)
      err2   = keras.layers.Lambda(lambda x:alpha[1]*x)(err2)
    
      # compute error (x_proj-x_input)**2 with full-1 mask
      x_proj3 = model_AE([x_proj,keras.layers.Lambda(lambda x:0.*x)(mask)])
      err3    = keras.layers.Subtract()([x_proj3,x_proj])
      err3    = keras.layers.Multiply()([err3,err3])
      err3    = keras.layers.Reshape((shape[1],shape[2],shape[3],1))(err3)
      err3    = keras.layers.GlobalAveragePooling3D()(err3)
      err3    = keras.layers.Reshape((1,))(err3)
      err3    = keras.layers.Lambda(lambda x:alpha[2]*x)(err3)

      err    = keras.layers.Add()([err1,err2])
      err    = keras.layers.Add()([err,err3])
      global_model_Grad_Masked  = keras.models.Model([x_input,mask],err)
    
      #global_model_Grad_Masked  = keras.models.Model([x_input,mask],[x_proj])
    
      if flagDisplay == 1:
          gradModel.summary()
          gradMaskModel.summary()
          global_model_Grad.summary()
          global_model_Grad_Masked.summary()
      
      return global_model_Grad,global_model_Grad_Masked

    for kk in range(0,len(flagProcess)):
        ###############################################################
        ## read dataset
        if flagProcess[kk] == 0:        
            if flagDataset == 0: ## MNIST
              (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
              
              x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
              x_test  = x_test.reshape((x_test.shape[0],x_train.shape[1],x_train.shape[2],1))
              
              dirSAVE = './MNIST/'
              genFilename = 'mnist_DINConvAE_v3_'
              flagloadOIData = 0

              meanTr     = np.mean(x_train)
              x_train    = x_train - meanTr
              x_test     = x_test  - meanTr
            
              # scale wrt std
              stdTr      = np.sqrt( np.mean( x_train**2 ) )
              x_train    = x_train / stdTr
              x_test     = x_test  / stdTr
            elif flagDataset == 1: ## FASHION MNIST

              (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

              x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
              x_test  = x_test.reshape((x_test.shape[0],x_train.shape[1],x_train.shape[2],1))
              dirSAVE = './MNIST/'
              genFilename = 'fashion_mnist_DINConvAE_'
              flagloadOIData = 0

              meanTr     = np.mean(x_train)
              x_train    = x_train - meanTr
              x_test     = x_test  - meanTr
            
              # scale wrt std
              stdTr      = np.sqrt( np.mean( x_train**2 ) )
              x_train    = x_train / stdTr
              x_test     = x_test  / stdTr

            elif flagDataset == 2: ## NATL60-METOP Dataset
              np.random.seed(100)
              thrMisData = 0.25
              indT       = np.arange(0,5)#np.arange(0,5)#np.arange(0,5)
              indN_Tr    = np.arange(0,415)#np.arange(0,35000)
              indN_Tt    = np.arange(650,800)
              
              if 1*0:
                  indT       = np.arange(2,3)#np.arange(0,5)#np.arange(0,5)
                  indN_Tr    = np.arange(0,800)#np.arange(0,600)#np.arange(0,10)#np.arange(0,35000)
                  indN_Tt    = np.arange(0,300)#np.arange(650,800)

              from netCDF4 import Dataset
              #fileTr = 'Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_20080601_20080731_Patch_032_032_005.nc'
              fileTr    = []
              fileTt    = []
              #fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080601_20080831_Patch_032_032_005.nc')
              #fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080601_20080630_Patch_064_064_005.nc')
              #fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080701_20080731_Patch_064_064_005.nc')
              #fileTr.append('/tmp/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080801_20080831_Patch_064_064_005.nc')
                            
              #fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080601_20080630_Patch_128_128_005.nc')
              #fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080701_20080731_Patch_128_128_005.nc')
              #fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080801_20080831_Patch_128_128_005.nc')

              if 1*0 :
                  fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080701_20080731_Patch_128_512_005.nc')
                  fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080601_20080630_Patch_128_512_005.nc')
                  fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080801_20080831_Patch_128_512_005.nc')

                  fileTt.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080701_20080731_Patch_128_512_005.nc')
              if 1*0:
                  fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_20080701_20080731_Patch_128_512_005.nc')
                  fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_20080601_20080630_Patch_128_512_005.nc')
                  fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_20080801_20080831_Patch_128_512_005.nc')
  
                  fileTt.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_20080701_20080731_Patch_128_512_005.nc')
              if 1*1:
                  thrMisData = 0.1
                  indT       = np.arange(0,11)#np.arange(0,5)#np.arange(0,5)
                  indN_Tr    = np.arange(0,200)#np.arange(0,800)#np.arange(0,600)#np.arange(0,35000)
                  indN_Tt    = np.arange(0,200)#np.arange(300,600)#np.arange(650,800)
                  SuffixOI   = '_OI_DT11Lx075Ly075Lt003'
                  
                  fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080601_20080630_Patch_128_512_011.nc')
                  #fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080701_20080731_Patch_128_512_011.nc')
                  
                  fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080801_20080831_Patch_128_512_011.nc')
                  fileTr.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080901_20080930_Patch_128_512_011.nc')
  
                  #fileTt.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080701_20080731_Patch_128_512_011.nc')
                  #fileTt.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080801_20080831_Patch_128_512_011.nc')
                  fileTt.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080701_20080731_Patch_128_512_011.nc')
                  #fileTt.append('Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080901_20080930_Patch_128_512_011.nc')

              for ii in range(0,len(fileTr)):
                  print(".... Load SST dataset (training data): "+fileTr[ii])
                  nc_data     = Dataset(fileTr[ii],'r')
              
                  print('.... # samples: %d '%nc_data.dimensions['N'].size)
                  if nc_data.dimensions['N'].size < indN_Tr[-1]:
                      x_train_ii    = nc_data['sst'][:,:,:,indT]
                      mask_train_ii = nc_data['mask'][:,:,:,indT]
                  else:
                      x_train_ii    = nc_data['sst'][indN_Tr,:,:,indT]
                      mask_train_ii = nc_data['mask'][indN_Tr,:,:,indT]
                  print('.... # loaded samples: %d '%x_train_ii.shape[0])
                      
                  # binary mask (remove non-binary labels due to re-gridding)
                  mask_train_ii     = (mask_train_ii > 0.5).astype('float')
                  
                  #x_train_ii    = nc_data['sst'][1000:6000,0:64:2,0:64:2,0]
                  #mask_train_ii = nc_data['mask'][1000:6000,0:64:2,0:64:2,0]
                  
                  nc_data.close()
              
                  if len(indT) == 1:
                      x_train_ii   = x_train_ii.reshape((x_train_ii.shape[0],x_train_ii.shape[1],x_train_ii.shape[2],1))
                      mask_train_ii = mask_train_ii.reshape((mask_train_ii.shape[0],mask_train_ii.shape[1],mask_train_ii.shape[2],1))
                                             
                  # load OI data
                  if flagloadOIData == 1:
                      print(".... Load OI SST dataset (training data): "+fileTr[ii].replace('.nc',SuffixOI+'.nc'))
                      nc_data       = Dataset(fileTr[ii].replace('.nc',SuffixOI+'.nc'),'r')
                      x_train_OI_ii = nc_data['sstOI'][:,:,:]
                      

                  # remove patch if no SST data
                  ss            = np.sum( np.sum( np.sum( x_train_ii < -100 , axis = -1) , axis = -1 ) , axis = -1)
                  ind           = np.where( ss == 0 )
                  
                  #print('... l = %d %d %d'%(ss.shape[0],x_train_ii.shape[0],len(ind[0])))
                  x_train_ii    = x_train_ii[ind[0],:,:,:]
                  mask_train_ii = mask_train_ii[ind[0],:,:,:]
                  if flagloadOIData == 1:
                      x_train_OI_ii = x_train_OI_ii[ind[0],:,:]
                  
                  rateMissDataTr_ii = np.sum( np.sum( np.sum( mask_train_ii , axis = -1) , axis = -1 ) , axis = -1)
                  rateMissDataTr_ii /= mask_train_ii.shape[1]*mask_train_ii.shape[2]*mask_train_ii.shape[3]
                  if 1*1:
                      ind        = np.where( rateMissDataTr_ii  >= thrMisData )              
                      x_train_ii    = x_train_ii[ind[0],:,:,:]
                      mask_train_ii = mask_train_ii[ind[0],:,:,:]                      
                      if flagloadOIData == 1:
                          x_train_OI_ii = x_train_OI_ii[ind[0],:,:]
                      
                  print('.... # remaining samples: %d '%x_train_ii.shape[0])
              
                  if ii == 0:
                      x_train    = np.copy(x_train_ii)
                      mask_train = np.copy(mask_train_ii)
                      if flagloadOIData == 1:
                          x_train_OI = np.copy(x_train_OI_ii)
                  else:
                      x_train    = np.concatenate((x_train,x_train_ii),axis=0)
                      mask_train = np.concatenate((mask_train,mask_train_ii),axis=0)
                      if flagloadOIData == 1:
                          x_train_OI = np.concatenate((x_train_OI,x_train_OI_ii),axis=0)
                                                      
              rateMissDataTr = np.sum( np.sum( np.sum( mask_train , axis = -1) , axis = -1 ) , axis = -1)
              rateMissDataTr /= mask_train.shape[1]*mask_train.shape[2]*mask_train.shape[3]
                  
              if NSampleTr <  x_train.shape[0] :                   
                  ind_rand = np.random.permutation(x_train.shape[0])

                  x_train    = x_train[ind_rand[0:NSampleTr],:,:,:]
                  mask_train = mask_train[ind_rand[0:NSampleTr],:,:,:]
                  if flagloadOIData == 1:
                      x_train_OI = x_train_OI[ind_rand[0:NSampleTr],:,:]
                      
              y_train    = np.ones((x_train.shape[0]))
              #mask_train = maskSet[:,:,:,:]
              #del sstSet
              if flagloadOIData:
                  print("....... # of training patches: %d/%d"%(x_train.shape[0],x_train_OI.shape[0]))
              else:
                  print("....... # of training patches: %d"%(x_train.shape[0]))
              
              if 1*1: 

                  print(".... Load SST dataset (test data): "+fileTt[0])              
                  nc_data     = Dataset(fileTt[0],'r')
                  if nc_data.dimensions['N'].size < indN_Tt[-1]:
                      x_test    = nc_data['sst'][:,:,:,indT]
                      mask_test = nc_data['mask'][:,:,:,indT]
                  else:
                      x_test    = nc_data['sst'][indN_Tt,:,:,indT]
                      mask_test = nc_data['mask'][indN_Tt,:,:,indT]
                  mask_test     = (mask_test > 0.5).astype('float')
                  
                  #x_test    = nc_data['sst'][7000:10000,0:64:2,0:64:2,0]
                  #mask_test = nc_data['mask'][7000:10000,0:64:2,0:64:2,0]
                  
                  nc_data.close()
                  if len(indT) == 1:
                      x_test    = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
                      mask_test = mask_test.reshape((mask_test.shape[0],mask_test.shape[1],mask_test.shape[2],1))

                  # load OI data
                  if flagloadOIData == 1:
                      print(".... Load OI SST dataset (test data): "+fileTt[0].replace('.nc',SuffixOI+'.nc'))
                      nc_data   = Dataset(fileTt[0].replace('.nc',SuffixOI+'.nc'),'r')
                      x_test_OI = nc_data['sstOI'][:,:,:]
                  
                   # remove patch if no SST data
                  ss        = np.sum( np.sum( np.sum( x_test < -100 , axis = -1) , axis = -1 ) , axis = -1)
                  ind       = np.where( ss == 0 )
                  x_test    = x_test[ind[0],:,:,:]
                  mask_test = mask_test[ind[0],:,:,:]
                  rateMissDataTt = np.sum( np.sum( np.sum( mask_test , axis = -1) , axis = -1 ) , axis = -1)
                  rateMissDataTt /= mask_test.shape[1]*mask_test.shape[2]*mask_test.shape[3]
                  
                  if flagloadOIData == 1:
                      x_test_OI    = x_test_OI[ind[0],:,:]
                                    
                  if 1*0:
                      ind       = np.where( rateMissDataTt >= thrMisData )              
                      x_test    = x_test[ind[0],:,:,:]
                      mask_test = mask_test[ind[0],:,:,:]
                  y_test    = np.ones((x_test.shape[0]))
              else:
                  Nt        = int(np.floor(x_train.shape[0]*0.25))
                  x_test    = np.copy(x_train[0:Nt,:,:,:])
                  mask_test = np.copy(mask_train[0:Nt,:,:,:])
                  y_test    = np.ones((x_test.shape[0]))
                 
                  x_train    = x_train[Nt+1::,:,:,:]
                  mask_train = mask_train[Nt+1::,:,:,:]
                  y_train    = np.ones((x_train.shape[0]))
                           
              if flagloadOIData:
                  print("....... # of test patches: %d /%d"%(x_test.shape[0],x_test_OI.shape[0]))
              else:
                  print("....... # of test patches: %d"%(x_test.shape[0]))

                    
              print("... mean Tr = %f"%(np.mean(x_train)))
              print("... mean Tt = %f"%(np.mean(x_test)))
                    
              print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
            
              dirSAVE = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/'
              
              if fileTr[0].find('Anomaly') == -1 :
                  genFilename = 'model_patchDataset_NATL60withMETOP_SST_'+str('%03d'%x_train.shape[1])+str('_%03d'%x_train.shape[2])+str('_%03d'%x_train.shape[3])
              else:
                genFilename = 'model_patchDataset_NATL60withMETOP_SSTAnomaly_'+str('%03d'%x_train.shape[1])+str('_%03d'%x_train.shape[2])+str('_%03d'%x_train.shape[3])
                
              print('....... Generic model filename: '+genFilename)
              
              meanTr     = np.mean(x_train)
              x_train    = x_train - meanTr
              x_test     = x_test  - meanTr
            
              if flagloadOIData:
                  x_train_OI    = x_train_OI - meanTr
                  x_test_OI     = x_test_OI  - meanTr

              # scale wrt std
              stdTr      = np.sqrt( np.mean( x_train**2 ) )
              x_train    = x_train / stdTr
              x_test     = x_test  / stdTr

              print('... Mean and std of training data: %f  -- %f'%(meanTr,stdTr))

              if flagloadOIData == 1:
                  x_train_OI    = x_train_OI / stdTr
                  x_test_OI     = x_test_OI  / stdTr

            if flagDataWindowing == 1:
                HannWindow = np.reshape(np.hanning(x_train.shape[2]),(x_train.shape[1],1)) * np.reshape(np.hanning(x_train.shape[1]),(x_train.shape[2],1)).transpose() 

                x_train = np.moveaxis(np.moveaxis(x_train,3,1) * np.tile(HannWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
                x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(HannWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
                print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
                print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

            elif flagDataWindowing == 2:
                EdgeWidth  = 4
                EdgeWindow = np.zeros((x_train.shape[1],x_train.shape[2]))
                EdgeWindow[EdgeWidth:x_train.shape[1]-EdgeWidth,EdgeWidth:x_train.shape[2]-EdgeWidth] = 1
                
                x_train = np.moveaxis(np.moveaxis(x_train,3,1) * np.tile(EdgeWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
                x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(EdgeWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)

                mask_train = np.moveaxis(np.moveaxis(mask_train,3,1) * np.tile(EdgeWindow,(mask_train.shape[0],x_train.shape[3],1,1)),1,3)
                mask_test  = np.moveaxis(np.moveaxis(mask_test,3,1) * np.tile(EdgeWindow,(mask_test.shape[0],x_test.shape[3],1,1)),1,3)
                print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
                print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
              
#            else:
#              mini = np.amin(x_train[:])
#              maxi = np.amax(x_train[:])
#              
#              x_train = (x_train - mini ) /(maxi-mini)
#              x_test  = (x_test - mini ) /(maxi-mini)
            
            print("... (after normalization) mean Tr = %f"%(np.mean(x_train)))
            print("... (after normalization) mean Tt = %f"%(np.mean(x_test)))
              
        ###############################################################
        ## generate missing data
        elif flagProcess[kk] == 1:
            
            if Wsquare > 0 :
                print("..... Generate missing data masks: %dx%dx%d "%(Nsquare,Wsquare,Wsquare))
                
                Wsquare = int(Wsquare)
                Nsquare = int(Nsquare)
                
                # random seed 
                np.random.seed(1)
                
                # generate missing data areas for training data
                x_train_missing = np.copy(x_train).astype(float)
                mask_train      = np.zeros((x_train.shape))
                mask_test       = np.zeros((x_test.shape))
                
                
                for ii in range(x_train.shape[0]):
                  # generate mask
                  mask   = np.ones((x_train.shape[1],x_train.shape[2],x_train.shape[3])).astype(float)
                  i_area = np.floor(np.random.uniform(Wsquare,x_train.shape[1]-Wsquare+1,Nsquare)).astype(int)
                  j_area = np.floor(np.random.uniform(Wsquare,x_train.shape[2]-Wsquare+1,Nsquare)).astype(int)
                  
                  for nn in range(Nsquare):
                    mask[i_area[nn]-Wsquare:i_area[nn]+Wsquare,j_area[nn]-Wsquare:j_area[nn]+Wsquare,:] = 0.
                    
                  # apply mask
                  x_train_missing[ii,:,:,:] *= mask
                  mask_train[ii,:,:,:]       = mask     
                  
                ## generate missing data areas for test data
                x_test_missing = np.copy(x_test).astype(float)
                
                for ii in range(x_test.shape[0]):
                  # generate mask
                  mask   = np.ones((x_test.shape[1],x_test.shape[2],x_train.shape[3])).astype(float)
                  i_area = np.floor(np.random.uniform(Wsquare,x_test.shape[1]-Wsquare+1,Nsquare)).astype(int)
                  j_area = np.floor(np.random.uniform(Wsquare,x_test.shape[2]-Wsquare+1,Nsquare)).astype(int)
                  
                  for nn in range(Nsquare):
                    mask[i_area[nn]-Wsquare:i_area[nn]+Wsquare,j_area[nn]-Wsquare:j_area[nn]+Wsquare,:] = 0.
                    
                  # apply mask
                  x_test_missing[ii,:,:,:] *= mask
                  mask_test[ii,:,:,:]      = mask     
            elif Wsquare == 0 :
                print("..... Use missing data masks from file ")
                Nsquare = 0
                x_train_missing = x_train * mask_train
                x_test_missing  = x_test  * mask_test
                 
#            elif Wsquare < 0 :
#                print("..... Use edge window mask ")
#                Nsquare = 0
#                x_train_missing = x_train * mask_train
#                x_test_missing  = np.moveaxis( np.tile(EdgeWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
#                x_test          = np.moveaxis( np.tile(EdgeWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
#                x_test_missing  = x_test  * mask_test

        ###############################################################
        ## define AE architecture
        elif flagProcess[kk] == 2:                    
            DimCAE = DimAE
            
            if flagAEType == 0: ## MLP-AE
              ## auto-encoder architecture (MLP)
              input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
             
              x          = keras.layers.Flatten()(input_data)
            
              encodeda   = keras.layers.Dense(DimAE, activation='linear')(x)
              encoder    = keras.models.Model([input_data,mask],encodeda)
            
              decoder_input = keras.layers.Input(shape=(DimAE,))
              decodedb      = keras.layers.Dense(x_train.shape[1]*x_train.shape[2]*x_train.shape[3])(decoder_input)
              decodedb      = keras.layers.Reshape((x_train.shape[1],x_train.shape[2],x_train.shape[3]))(decodedb)
              decoder       = keras.models.Model(decoder_input,decodedb)
            
              encoder.summary()
              decoder.summary()
            
              input_data = keras.layers.Input(shape=(x_train.shape[1]*x_train.shape[2]*x_train.shape[3],))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              x          = decoder(encoder([input_data,mask]))
              model_AE     = keras.models.Model([input_data,mask],x)
            
              model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-2))
              model_AE.summary()

              ## auto-encoder architecture (MLP)
              if 1*0:
                  input_layer = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
                
                  x          = keras.layers.Flatten()(input_layer)
                
                  x          = keras.layers.Dense(6*DimAE, activation='relu')(x)
                  #x          = keras.layers.Dropout(dropout)(x)
                  x          = keras.layers.Dense(2*DimAE, activation='relu')(x)
                  #x          = keras.layers.Dropout(dropout)(x)
                  #x          = keras.layers.BatchNormalization()(x)
                  #x          = keras.layers.Dense(3, activation='relu')(x)
                  encodeda   = keras.layers.Dense(DimAE, activation='linear')(x)
                  encoder    = keras.models.Model(input_layer,encodeda)
                
                  decoder_input = keras.layers.Input(shape=(DimAE,))
                  x             = keras.layers.Dense(10*DimAE, activation='relu')(decoder_input)
                  #x             = keras.layers.BatchNormalization()(x)
                  #x             = keras.layers.Dropout(dropout)(x)
                  #x             = keras.layers.Dense(3, activation='relu')(x)
                  x             = keras.layers.Dense(20*DimAE, activation='relu')(x)
                  #x             = keras.layers.Dropout(dropout)(x)
                  #x             = keras.layers.Dense(20*DimAE, activation='relu')(x)
                  decodedb      = keras.layers.Dense(x_train.shape[1]*x_train.shape[2]*x_train.shape[3])(x)
                  decodedb      = keras.layers.Reshape((x_train.shape[1],x_train.shape[2],x_train.shape[3]))(decodedb)
                  decoder       = keras.models.Model(decoder_input,decodedb)
                
                  encoder.summary()
                  decoder.summary()
                
                  input_data = keras.layers.Input(shape=(x_train.shape[1]*x_train.shape[2]*x_train.shape[3],))
                  x          = decoder(encoder(input_layer))
                  model_AE     = keras.models.Model(input_layer,x)
                
                  model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-2))
                  model_AE.summary()
            elif flagAEType == 1: ## Conv-AE
              input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
             
              #x = keras.layers.Reshape((x_train.shape[1],x_train.shape[2],1))(input_layer)
              x = keras.layers.Conv2D(6*DimAE,(x_train.shape[1],x_train.shape[2]),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(input_data)
              x = keras.layers.Conv2D(2*DimAE,(1,1),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              encoder    = keras.models.Model([input_data,mask],x)
            
            
              decoder_input = keras.layers.Input(shape=(1,1,DimAE))
            
              x = keras.layers.Conv2DTranspose(2*DimAE,(x_train.shape[1],x_train.shape[2]),strides=(x_train.shape[1],x_train.shape[2]),padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
              for kk in range(0,2):
                dx = keras.layers.Conv2D(5*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
                dx = keras.layers.Conv2D(2*DimAE,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                x  = keras.layers.Add()([x,dx])
              #x = keras.layers.Dropout(dropout)(x)
            
              x = keras.layers.Conv2D(x_train.shape[3],(1,1),activation='linear', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              #x = keras.layers.Reshape((x_train.shape[1],x_train.shape[2]))(x)
            
              decoder       = keras.models.Model(decoder_input,x)
            
              encoder.summary()
              decoder.summary()
            
              input_data = keras.layers.Input(shape=(x_train.shape[1]*x_train.shape[2]*x_train.shape[3],))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              x          = decoder(encoder([input_data,mask]))
              model_AE     = keras.models.Model([input_data,mask],x)
             
              model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
              model_AE.summary()
            elif flagAEType == 2: ## Conv-AE
              
              Wpool_i = np.floor(  (np.floor((x_train.shape[1]-2)/2)-2)/2 ).astype(int) 
              Wpool_j = np.floor(  (np.floor((x_train.shape[2]-2)/2)-2)/2 ).astype(int)
              
              input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            
              #x = keras.layers.Reshape((x_train.shape[1],x_train.shape[2],1))(input_layer)
              x = keras.layers.Conv2D(DimAE,(3,3),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(input_data)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(4*DimAE,(Wpool_i,Wpool_j),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              encoder    = keras.models.Model([input_data,mask],x)
            
            
              decoder_input = keras.layers.Input(shape=(1,1,DimAE))
            
              x = keras.layers.Conv2DTranspose(1*DimAE,(x_train.shape[1],x_train.shape[2]),strides=(x_train.shape[1],x_train.shape[2]),padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
              x = keras.layers.Dropout(dropout)(x)
              for kk in range(0,2):
                dx = keras.layers.Conv2D(5*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
                dx = keras.layers.Dropout(dropout)(dx)
                dx = keras.layers.Conv2D(1*DimAE,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                x  = keras.layers.Add()([x,dx])
              #x = keras.layers.Dropout(dropout)(x)
            
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2D(x_train.shape[3],(1,1),activation='linear', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              #x = keras.layers.Reshape((x_train.shape[1],x_train.shape[2]))(x)
            
              decoder       = keras.models.Model(decoder_input,x)
            
              encoder.summary()
              decoder.summary()
            
              input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              x          = decoder(encoder([input_data,mask]))
              model_AE   = keras.models.Model([input_data,mask],x)
            
              model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
              model_AE.summary()
            elif flagAEType == 3: ## Conv-AE for SST case-study
  
              Wpool_i = np.floor(  (np.floor((x_train.shape[1]-2)/2)-2)/2 ).astype(int) 
              Wpool_j = np.floor(  (np.floor((x_train.shape[2]-2)/2)-2)/2 ).astype(int)
              
              input_data  = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask        = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            
              x = keras.layers.Conv2D(DimAE,(3,3),activation='relu', padding='valid',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(input_data)                    
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(4*DimAE,(Wpool_i,Wpool_j),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              encoder    = keras.models.Model([input_data,mask],x)
                       
              decoder_input = keras.layers.Input(shape=(1,1,DimAE))
            
              x = keras.layers.Conv2DTranspose(4*DimAE,(int(x_train.shape[1]/2),int(x_train.shape[2]/2)),strides=(int(x_train.shape[1]/2),int(x_train.shape[2]/2)),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2DTranspose(2*DimAE,(4,4),strides=(2,2),activation='linear',use_bias=False,padding='valid',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              if 1*1 :
                  x = keras.layers.Dropout(dropout)(x)
                  for kk in range(0,2):
                    dx = keras.layers.Conv2D(4*DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
                    dx = keras.layers.Dropout(dropout)(dx)
                    dx = keras.layers.Conv2D(2*DimAE,(3,3), padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                    dx = keras.layers.Dropout(dropout)(dx)
                    x  = keras.layers.Add()([x,dx])
              #x = keras.layers.Dropout(dropout)(x)
            
              x = keras.layers.Conv2D(x_train.shape[3],(3,3),activation='linear', padding='valid',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              if 1*0: 
                  x = keras.layers.Conv2DTranspose(2*DimAE,(int(x_train.shape[1]/2),int(x_train.shape[2]/2)),strides=(int(x_train.shape[1]/2),int(x_train.shape[2]/2)),padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
                  x = keras.layers.Dropout(dropout)(x)
                  x = keras.layers.Conv2DTranspose(2*DimAE,(2,2),strides=(2,2),padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
                  x = keras.layers.Dropout(dropout)(x)
                  for kk in range(0,2):
                    dx = keras.layers.Conv2D(5*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
                    dx = keras.layers.Conv2D(2*DimAE,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                    x  = keras.layers.Add()([x,dx])
                  #x = keras.layers.Dropout(dropout)(x)
                
                  x = keras.layers.Dropout(dropout)(x)
                  x = keras.layers.Conv2D(x_train.shape[3],(1,1),activation='linear', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              #x = keras.layers.Reshape((x_train.shape[1],x_train.shape[2]))(x)
            
              decoder       = keras.models.Model(decoder_input,x)
            
              encoder.summary()
              decoder.summary()
            
              input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              x          = decoder(encoder([input_data,mask]))
              model_AE   = keras.models.Model([input_data,mask],x)
            
              model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
              model_AE.summary()
              
            elif flagAEType == 4: ## Conv-AE for SST case-study §64x64)  
              input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask        = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              x = keras.layers.Conv2D(DimAE,(3,3),activation='relu', padding='valid',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(input_data)                    
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(4*DimAE,(3,3),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(8*DimAE,(6,6),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              if 1*0:
                  x = keras.layers.Conv2D(32,(5,5),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(input_layer)                    
                  x = keras.layers.Dropout(dropout)(x)
                  x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
                  x = keras.layers.Conv2D(64,(5,5),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
                  x = keras.layers.Dropout(dropout)(x)
                  x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
                  x = keras.layers.Conv2D(128,(5,5),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
                  x = keras.layers.Dropout(dropout)(x)
                  #x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
                  x = keras.layers.Conv2D(256,(5,5),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
                  x = keras.layers.Dropout(dropout)(x)
                  #x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
                  x = keras.layers.Conv2D(DimAE,(5,5),activation='linear', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              encoder    = keras.models.Model([input_data,mask],x)
                       
              decoder_input = keras.layers.Input(shape=(1,1,DimAE))
            
              x = keras.layers.Conv2DTranspose(256,(8,8),strides=(8,8),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2DTranspose(128,(3,3),strides=(2,2),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2DTranspose(64,(3,3),strides=(2,2),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2DTranspose(20,(3,3),strides=(2,2),use_bias=False,activation='linear',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              
              #x = keras.layers.Conv2D(x_train.shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              if 1*1:
                  x = keras.layers.Dropout(dropout)(x)
                  for kk in range(0,3):
                    dx = keras.layers.Conv2D(40,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
                    dx = keras.layers.Dropout(dropout)(dx)
                    dx = keras.layers.Conv2D(20,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                    dx = keras.layers.Dropout(dropout)(dx)
                    x  = keras.layers.Add()([x,dx])
                
              x = keras.layers.Conv2D(x_train.shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              decoder       = keras.models.Model(decoder_input,x)
                        
              encoder.summary()
              decoder.summary()
            
              input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              x          = decoder(encoder([input_data,mask]))
              model_AE   = keras.models.Model([input_data,mask],x)
            
              model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
              model_AE.summary()

            elif flagAEType == 5: ## Conv-AE for SST case-study
  
              Wpool_i = np.floor(  (np.floor((x_train.shape[1]-2)/2)-2)/2 ).astype(int) 
              Wpool_j = np.floor(  (np.floor((x_train.shape[2]-2)/2)-2)/2 ).astype(int)
              
              input_layer = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            
              x = keras.layers.Conv2D(DimAE,(3,3),activation='relu', padding='valid',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(input_layer)                    
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(DimAE,(Wpool_i,Wpool_j),activation='linear', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              #x = keras.layers.Dropout(dropout)(x)
              #x = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              encoder    = keras.models.Model(input_layer,x)
                       
              decoder_input = keras.layers.Input(shape=(1,1,DimAE))
            
              x = keras.layers.Conv2DTranspose(4*DimAE,(int(x_train.shape[1]/2),int(x_train.shape[2]/2)),strides=(int(x_train.shape[1]/2),int(x_train.shape[2]/2)),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2DTranspose(2*DimAE,(4,4),strides=(2,2),activation='linear',use_bias=False,padding='valid',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              if 1*1 :
                  x = keras.layers.Dropout(dropout)(x)
                  for kk in range(0,2):
                    dx = keras.layers.Conv2D(4*DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
                    dx = keras.layers.Dropout(dropout)(dx)
                    dx = keras.layers.Conv2D(2*DimAE,(3,3), padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                    dx = keras.layers.Dropout(dropout)(dx)
                    x  = keras.layers.Add()([x,dx])
              #x = keras.layers.Dropout(dropout)(x)
            
              x = keras.layers.Conv2D(x_train.shape[3],(3,3),activation='linear', padding='valid',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              if 1*0: 
                  x = keras.layers.Conv2DTranspose(2*DimAE,(int(x_train.shape[1]/2),int(x_train.shape[2]/2)),strides=(int(x_train.shape[1]/2),int(x_train.shape[2]/2)),padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
                  x = keras.layers.Dropout(dropout)(x)
                  x = keras.layers.Conv2DTranspose(2*DimAE,(2,2),strides=(2,2),padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
                  x = keras.layers.Dropout(dropout)(x)
                  for kk in range(0,2):
                    dx = keras.layers.Conv2D(5*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
                    dx = keras.layers.Conv2D(2*DimAE,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                    x  = keras.layers.Add()([x,dx])
                  #x = keras.layers.Dropout(dropout)(x)
                
                  x = keras.layers.Dropout(dropout)(x)
                  x = keras.layers.Conv2D(x_train.shape[3],(1,1),activation='linear', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              #x = keras.layers.Reshape((x_train.shape[1],x_train.shape[2]))(x)
            
              decoder       = keras.models.Model(decoder_input,x)            
              encoder.summary()
              decoder.summary()
            
              input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              x          = decoder(encoder([input_data,mask]))
              model_AE   = keras.models.Model([input_data,mask],x)
            
              model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
              model_AE.summary()
            elif flagAEType == 6: ## Conv-AE for SST case-study §64x64)  
              input_layer = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))

              x = keras.layers.Conv2D(DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(input_layer)                    
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(4*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(8*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(16*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              encoder    = keras.models.Model([input_layer,mask],x)
                                   
              decoder_input = keras.layers.Input(shape=(int(np.floor(x_train.shape[1]/32)),int(np.floor(x_train.shape[2]/32)),DimAE))            
              x = keras.layers.Conv2DTranspose(256,(16,16),strides=(16,16),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2DTranspose(64,(3,3),strides=(2,2),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              #x = keras.layers.Conv2DTranspose(32,(3,3),strides=(2,2),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              #x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2D(32,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              for kk in range(0,2):
                  dx = keras.layers.Conv2D(64,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
                  dx = keras.layers.Dropout(dropout)(dx)
                  dx = keras.layers.Conv2D(32,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                  dx = keras.layers.Dropout(dropout)(dx)
                  x  = keras.layers.Add()([x,dx])

              x = keras.layers.Conv2D(x_train.shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              decoder       = keras.models.Model(decoder_input,x)
                        
              encoder.summary()
              decoder.summary()
            
              input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              x          = decoder(encoder([input_data,mask]))
              model_AE   = keras.models.Model([input_data,mask],x)
            
              model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
              model_AE.summary()
              if DimCAE > x_train.shape[0] :
                  DimCAE = DimAE
              
            elif flagAEType == 7: ## Energy function of the type ||x(p)-f(x(q, q<>p))||

              # Constraints on kernel to to zeros
              # a specific position
              class Constraint_Zero(Constraint):
                  def __init__(self, position,kernel_shape,dw):
                      self.position = position
                      mask_array    = np.ones((kernel_shape))
                      mask_array[position[0]-dw:position[0]+dw+1,position[1]-dw:position[1]+dw+1,:,:] = 0.0
                      #mask_array[position[0],position[1]] = 0.0
                      #mask_array[position[0]-dw:position[0]+dw,position[1]-dw:position[1]+dw] = 0.0
                                
                      self.mask = K.variable(value=mask_array, dtype='float32', name='mask') 
                        
                      print(self.mask.shape)
                  def __call__(self, w):
                        
                      print(self.mask.shape)
                      new_w = w * self.mask
                      
                      return new_w
                    
              WFilter       = 11#
              NbResUnit     = 10#3#
              dW            = 0
              flagdownScale = 1 #: 0: only HR scale, 1 : only LR, 2 : HR + LR , 2 : MR, HR + LR annd LR,
              scaleLR       = 2**2
              NbFilter      = 1*DimAE#20*DimAE
              flagSRResNet  = 0
              
              genFilename = genFilename+str('_dW%03d'%dW)+str('WFilter%03d_'%WFilter)+str('NFilter%03d_'%NbFilter)+str('RU%03d_'%NbResUnit)
                  
              if flagdownScale == 0 :
                  genFilename = genFilename+str('HR')
              elif flagdownScale == 1 :
                  if flagSRResNet == 0 :
                      genFilename = genFilename+str('LR%03dwoSR'%scaleLR)
                  else:
                      genFilename = genFilename+str('LR%03dSR'%scaleLR)
              elif flagdownScale == 2 :
                  genFilename = genFilename+str('LRHR%03d'%scaleLR)
              elif flagdownScale == 3 :
                  genFilename = genFilename+str('MR%03d'%scaleLR)
       

              input_layer = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              
              if flagUseMaskinEncoder == 1:
                  dmask   = keras.layers.Lambda(lambda x: 0.2*x - 0.1)(mask)
                  
                  for jj in range(0,6):
                      dx    = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dmask)                    
                      dx    = keras.layers.Conv2D(1,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)                    
                      dmask = keras.layers.Add()([dmask,dx])
                      
                  x0       = keras.layers.Concatenate(axis=-1)([input_layer,dmask])
              else:
                  x0  = keras.layers.Lambda(lambda x: 1. * x)(input_layer)
                  
              # coarse scale
              if flagdownScale > 0 :
                  if flagUseMaskinEncoder == 1:
                      x = keras.layers.AveragePooling2D((scaleLR,scaleLR), padding='valid')(x0)
                      
                      x = keras.layers.Conv2D(NbFilter,(WFilter,WFilter),activation='relu', 
                                  padding='same',use_bias=False,
                                  kernel_regularizer=keras.regularizers.l2(wl2),
                                  kernel_constraint=Constraint_Zero((int(WFilter/2),int(WFilter/2)),(WFilter,WFilter,2*x_train.shape[3],NbFilter),dW))(x)
                  else:
                      x = keras.layers.AveragePooling2D((scaleLR,scaleLR), padding='valid')(x0)

                      x = keras.layers.Conv2D(NbFilter,(WFilter,WFilter),activation='relu', 
                                      padding='same',use_bias=False,
                                      kernel_regularizer=keras.regularizers.l2(wl2),
                                      kernel_constraint=Constraint_Zero((int(WFilter/2),int(WFilter/2)),(WFilter,WFilter,x_train.shape[3],NbFilter),dW))(x)
                  x  = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    
                  # registration/evolution in feature space
                  scale = 0.1
                  for kk in range(0,NbResUnit):
                      if 1*1 :
                          dx      = keras.layers.Conv2D(5*DimAE,(1,1),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
                      else:
                          dx  = keras.layers.Lambda(lambda x: scale * x)(x)
                        
                      dx_lin  = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                      dx1 = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                      dx2 = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                        
                      dx1 = keras.layers.Multiply()([dx1,dx2])
                      
                      dx  = keras.layers.Add()([dx1,dx_lin])
                      #dx  = keras.layers.Lambda(lambda x: scale * x)(dx)
                      dx  = keras.layers.Activation('tanh')(dx)
                      #dx  = keras.layers.Lambda(lambda x: (1./scale) * x)(dx)
                      x  = keras.layers.Add()([x,dx])
    
                  x  = keras.layers.Conv2D(x_train.shape[3],(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)                  
                  x1 = keras.layers.Conv2DTranspose(x_train.shape[3],(scaleLR,scaleLR),strides=(scaleLR,scaleLR),use_bias=False,activation='linear',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
                  
                  if flagSRResNet == 1: ## postprocessing: super-resolution-like block
                      #x1  = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)                  
                      x1  = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)                                       
                      
                      scale = 0.1
                      for kk in range(0,NbResUnit):
                          if 1*1 :
                              dx      = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)
                          else:
                              dx  = keras.layers.Lambda(lambda x: scale * x)(x1)
                            
                          dx_lin  = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                          dx1 = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                          dx2 = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                            
                          dx1 = keras.layers.Multiply()([dx1,dx2])
                          
                          dx  = keras.layers.Add()([dx1,dx_lin])
                          #dx  = keras.layers.Lambda(lambda x: scale * x)(dx)
                          dx  = keras.layers.Activation('tanh')(dx_lin)
                          #dx  = keras.layers.Lambda(lambda x: (1./scale) * x)(dx)
                          x1  = keras.layers.Add()([x1,dx])
                      x1  = keras.layers.Conv2D(x_train.shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)                  
                  else:
                      x1  = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)                  
                      x1  = keras.layers.Conv2D(x_train.shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)                  
                      
              # fine scale
              if flagUseMaskinEncoder == 1:
                  x = keras.layers.Conv2D(NbFilter,(WFilter,WFilter),activation='relu', 
                                      padding='same',use_bias=False,
                                      kernel_regularizer=keras.regularizers.l2(wl2),
                                      kernel_constraint=Constraint_Zero((int(WFilter/2),int(WFilter/2)),(WFilter,WFilter,2*x_train.shape[3],NbFilter),dW))(x0)
              else:
                  x = keras.layers.Conv2D(NbFilter,(WFilter,WFilter),activation='relu', 
                                      padding='same',use_bias=False,
                                      kernel_regularizer=keras.regularizers.l2(wl2),
                                      kernel_constraint=Constraint_Zero((int(WFilter/2),int(WFilter/2)),(WFilter,WFilter,x_train.shape[3],NbFilter),dW))(x0)

              if flagdownScale > 0 :
                  x  = keras.layers.Concatenate(axis=-1)([x,x1])
              x  = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    
              for kk in range(0,NbResUnit):
                  if 1*1 :
                      dx      = keras.layers.Conv2D(5*DimAE,(1,1),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
                  else:
                      dx  = keras.layers.Lambda(lambda x: scale * x)(x)
                    
                  dx_lin  = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                  dx1 = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                  dx2 = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                    
                  dx1 = keras.layers.Multiply()([dx1,dx2])
                  
                  dx  = keras.layers.Add()([dx1,dx_lin])
                  #dx  = keras.layers.Lambda(lambda x: scale * x)(dx)
                  dx  = keras.layers.Activation('tanh')(dx)
                  #dx  = keras.layers.Lambda(lambda x: (1./scale) * x)(dx)
                  x  = keras.layers.Add()([x,dx])

              x  = keras.layers.Conv2D(x_train.shape[3],(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
                              
              if flagdownScale == 0:
                  encoder    = keras.models.Model([input_layer,mask],x)
              elif flagdownScale == 1:
                  encoder    = keras.models.Model([input_layer,mask],x1)
              elif flagdownScale == 2:
                  x          = keras.layers.Add()([x,x1]) 
                  encoder    = keras.models.Model([input_layer,mask],x)
              elif flagdownScale == 3:
                  x          = keras.layers.Add()([x,x1]) 
                  encoder    = keras.models.Model([input_layer,mask],[x1,x])
             
              if flagdownScale == 3:
                  decoder_input1 = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))            
                  decoder_input2 = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))            
                  x1  = keras.layers.Lambda(lambda x: 1. * x)(decoder_input1)
                  x2  = keras.layers.Lambda(lambda x: 1. * x)(decoder_input2)
                  decoder       = keras.models.Model([decoder_input1,decoder_input2],[x1,x2])
              else:
                  decoder_input = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))            
                  x  = keras.layers.Lambda(lambda x: 1. * x)(decoder_input)
                  decoder       = keras.models.Model(decoder_input,x)
                        
              encoder.summary()
              decoder.summary()
            
              if flagdownScale < 3:
                  input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
                  mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    
                  x          = decoder(encoder([input_data,mask]))
                  model_AE   = keras.models.Model([input_data,mask],x)              
              elif flagdownScale == 3:
                  flag_MultiScaleAEModel = 1
                  
                  input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
                  mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
                  x,xLR      = encoder([input_data,mask])
                  
                  model_AE    = keras.models.Model([input_data,mask],x)
                  model_AE_MR = keras.models.Model([input_data,mask],[x,xLR])
            
              model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
              model_AE.summary()

              DimCAE = DimAE
            elif flagAEType == 8: ## Conv-AE for SST case-study §64x64)  
              input_layer = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask        = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              
              if flagUseMaskinEncoder == 1:
                  dmask   = keras.layers.Lambda(lambda x: 0.2*x - 0.1)(mask)
                  
                  for jj in range(0,6):
                      dx    = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dmask)                    
                      dx    = keras.layers.Conv2D(1,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)                    
                      dmask = keras.layers.Add()([dmask,dx])
                      
                  x       = keras.layers.Concatenate(axis=-1)([input_layer,dmask])
                  x       = keras.layers.Conv2D(DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)                    
              else:
                  x       = keras.layers.Conv2D(DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(input_layer)                    

              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
              x = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              encoder    = keras.models.Model([input_layer,mask],x)
                                   
              decoder_input = keras.layers.Input(shape=(int(np.floor(x_train.shape[1]/32)),int(np.floor(x_train.shape[2]/32)),DimAE))            
              
              x = keras.layers.Conv2DTranspose(64,(16,16),strides=(16,16),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
              x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2DTranspose(32,(3,3),strides=(2,2),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              x = keras.layers.Dropout(dropout)(x)
              #x = keras.layers.Conv2DTranspose(32,(3,3),strides=(2,2),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              #x = keras.layers.Dropout(dropout)(x)
              x = keras.layers.Conv2D(16,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              
              x = keras.layers.Conv2D(x_train.shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
              decoder       = keras.models.Model(decoder_input,x)
                        
              encoder.summary()
              decoder.summary()
            
              input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              x          = decoder(encoder([input_data,mask]))
              model_AE   = keras.models.Model([input_data,mask],x)
            
              model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
              model_AE.summary()
              DimCAE = DimAE * int(np.floor(x_train.shape[1]/32)) * int(np.floor(x_train.shape[2]/32))
              if DimCAE > x_train.shape[0] :
                  DimCAE = DimAE

        ###############################################################
        ## define classifier architecture for performance evaluation
        elif flagProcess[kk] == 3:
            num_classes = (np.max(y_train)+1).astype(int)
            
            classifier = keras.Sequential()
            classifier.add(keras.layers.Dense(32,activation='relu', input_shape=(DimAE,)))
            classifier.add(keras.layers.Dense(64,activation='relu'))
            classifier.add(keras.layers.Dense(num_classes, activation='softmax'))
            
            classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            classifier.summary()

        ###############################################################
        ## train Conv-AE
        elif flagProcess[kk] == 4:        

            # PCA decomposition for comparison
            pca              = decomposition.PCA(DimCAE)
            pca.fit(np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])))
            
            print(x_train.shape)
            print(x_test.shape)
            
            
            rec_PCA_Tt       = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
            rec_PCA_Tt[:,DimCAE:] = 0.
            rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
            mse_PCA_Tt       = np.mean( (rec_PCA_Tt - x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))**2 )
            var_Tt           = np.mean( (x_test-np.mean(x_train,axis=0))** 2 )
            exp_var_PCA_Tt   = 1. - mse_PCA_Tt / var_Tt
            
            print(".......... PCA Dim = %d"%(DimCAE))
            print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1]))
            print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))

            print("..... Regularization parameters: dropout = %.3f, wl2 = %.2E"%(dropout,wl2))
            
            # model compilation
            # model fit
            if flagDataset == 2 :
                NbProjection   = [0,0,2,2,5,5,10,15,14]#[0,0,0,0,0,0]#[5,5,5,5,5]##
                lrUpdate       = [1e-3,1e-4,1e-3,1e-4,1e-5,1e-6,1e-6,1e-5,1e-6]
                lrUpdate       = [1e-4,1e-5,1e-4,1e-5,1e-5,1e-6,1e-6,1e-5,1e-6]
                IterUpdate     = [0,3,10,15,20,25,30,35,40]#[0,2,4,6,9,15]
                if flagOptimMethod == 1 :
                    NbProjection   = [0,0,0,0,0,0,0,0]#[0,2,1,1,0,0]#[0,2,2,2,2,2]#[5,5,5,5,5]##
                    NbGradIter     = [0,1,2,5,5,8,12,12]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
                    IterUpdate     = [0,3,10,15,20,30,35,40]#[0,2,4,6,9,15]
                    lrUpdate       = [1e-3,1e-5,1e-4,1e-5,1e-5,1e-5,1e-6,1e-5,1e-6]
                #IterUpdate     = [0,3,6,10,15,20,25,30]#[0,2,4,6,9,15]
                val_split      = 0.1
            else:
                NbProjection   = [0,5,5,10,15,15]#[0,0,0,0,0,0]#[5,5,5,5,5]##
                NbProjection   = [0,0,5,10,15,15]#[0,0,0,0,0,0]#[5,5,5,5,5]##
                if flagOptimMethod == 1 :
                    NbProjection   = [0,0,0,0,0,0]#[0,2,1,1,0,0]#[0,2,2,2,2,2]#[5,5,5,5,5]##
                    NbGradIter     = [0,5,5,10,12,14]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
                IterUpdate     = [0,3,6,10,15,20]#[0,2,4,6,9,15]
                lrUpdate       = [1e-4,1e-5,1e-5,1e-6,1e-7,1e-7]
                #lrUpdate       = [1e-4,1e-4,1e-5,1e-5,1e-4,1e-5]
                #lrUpdate       = [1e-4,1e-4,1e-5,1e-6,1e-6,1e-7]
                val_split      = 0.1
            
            flagLoadModelAE = 0
            fileAEModelInit = './MNIST/mnist_DINConvAE_AE02N03W04_Nproj10_Encoder_iter015.mod'#'./MNIST/mnist_DINConvAE_AE02N30W02_Nproj05_Encoder_iter006.mod'#'./MNIST/mnist_DINConvAE_AE02N30W02_Nproj10_Encoder_iter011.mod'
            
            fileAEModelInit = './MNIST/mnist_DINConvAE_AETRwoMissingData02N06W04_Nproj10_Encoder_iter015.mod'
            fileAEModelInit = './MNIST/mnist_DINConvAE_AE02N30W02_Nproj10_Encoder_iter015.mod'
            fileAEModelInit = './MNIST/mnist_DINConvAE_AETRwoMissingData02N30W02_Nproj10_Encoder_iter015.mod'
            #fileAEModelInit = './MNIST/fashion_mnist_DINConvAE_AE02N20W02_Nproj10_Encoder_iter015.mod'
            #fileAEModelInit = './MNIST/mnist_DINConvAE_GradAE02_00_D20N06W04_Nproj01_Grad15_Encoder_iter008.mod'
            #fileAEModelInit = './MNIST/mnist_DINConvAE_GradAE02_00D20N06W04_Nproj01_Grad10_Encoder_iter005.mod'
            #fileAEModelInit = './MNIST/mnist_DINConvAE_AE03N30W02_Nproj15_Encoder_iter018.mod'
            fileAEModelInit = './MNIST/mnist_DINConvAE_GradAETRwoMissingData02_00D20N30W02_Nproj05_Grad05_Encoder_iter005.mod'
            
            fileAEModelInit = './MNIST/mnist_DINConvAE_v3__Alpha102GradAE02_00_D20N06W04_Nproj00_Grad10_Encoder_iter012.mod'
            #fileAEModelInit = './MNIST/mnist_DINConvAE_v3__Alpha100_AE02D20N03W04_Nproj10_Encoder_iter014.mod'
            
            
            if 1*0:
                fileAEModelInit = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_dW000WFilter005_NFilter020_RU005_LR004_Alpha100_AE07D20N00W00_Nproj05_Encoder_iter018.mod'
                fileAEModelInit = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_dW000WFilter005_NFilter020_RU005_LR004woSR_Alpha100_AE07D20N00W00_Nproj10_Encoder_iter029.mod'
                fileAEModelInit = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_dW000WFilter003_NFilter020_RU003_LR004woSR_Alpha100_AE07D20N00W00_Nproj15_Encoder_iter043.mod'
                #fileAEModelInit = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_Alpha100_AE08D20N00W00_Nproj10_Encoder_iter029.mod'
                fileAEModelInit = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_dW000WFilter009_NFilter020_RU003_LR004woSR_Alpha100_AE07D20N00W00_Nproj05_Encoder_iter029.mod'
                fileAEModelInit = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_Alpha101GradAE08_00_D40N00W00_Nproj00_Grad05_Encoder_iter021.mod'
                
            iterInit        = 0
            if flagLoadModelAE > 0 :
                iterInit        = 13
            IterTrainAE     = 0

            IterUpdateInit = 10000
            
            ## initialization
            x_train_init = np.copy(x_train_missing)
            x_test_init  = np.copy(x_test_missing)

            comptUpdate = 0
            if flagLoadModelAE > 0 :
                print('.................. Load Encoder/Decoder '+fileAEModelInit)
                encoder.load_weights(fileAEModelInit)
                decoder.load_weights(fileAEModelInit.replace('Encoder','Decoder'))

                if flagOptimMethod == 0:
                    comptUpdate = 3
                    NBProjCurrent = NbProjection[comptUpdate-1]
                    print("..... Initialize number of projections in DINCOnvAE model # %d"%(NbProjection[comptUpdate-1]))
                    global_model_FP,global_model_FP_Masked = define_DINConvAE(NbProjection[comptUpdate-1],model_AE,x_train.shape,alpha)
                    if flagTrOuputWOMissingData == 1:
                        global_model_FP.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate-1]))
                    else:
                        global_model_FP_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate-1]))
                elif flagOptimMethod == 1:
                    for layer in encoder.layers:
                        layer.trainable = True
                    for layer in decoder.layers:
                        layer.trainable = True
                    
                    comptUpdate   = 4
                    NBProjCurrent = NbProjection[comptUpdate-1]
                    NBGradCurrent = NbGradIter[comptUpdate-1]
                    print("..... Initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection[comptUpdate-1],NbGradIter[comptUpdate-1]))
                    gradModel,gradMaskModel =  define_GradModel(model_AE,x_train.shape,flagGradModel)
                    
                    if flagLoadModelAE == 2:
                        gradMaskModel.load_weights(fileAEModelInit.replace('Encoder','GradMaskModel'))
                        gradModel.load_weights(fileAEModelInit.replace('Encoder','GradModel'))

                    global_model_Grad,global_model_Grad_Masked = define_GradDINConvAE(NbProjection[comptUpdate-1],NbGradIter[comptUpdate-1],model_AE,x_train.shape,gradModel,gradMaskModel,flagGradModel)
                    if flagTrOuputWOMissingData == 1:
                        global_model_Grad.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate-1]))
                    else:
                        global_model_Grad_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate-1]))
            else:
                if flagOptimMethod == 1:
                    gradModel,gradMaskModel =  define_GradModel(model_AE,x_train.shape,flagGradModel)
                
            print("..... Start learning AE model %d FP/Grad %d"%(flagAEType,flagOptimMethod))
            for iter in range(iterInit,Niter):
                if iter == IterUpdate[comptUpdate]:
                    if flagOptimMethod == 0:
                        # update DINConvAE model
                        NBProjCurrent = NbProjection[comptUpdate]
                        print("..... Update/initialize number of projections in DINCOnvAE model # %d"%(NbProjection[comptUpdate]))
                        global_model_FP,global_model_FP_Masked = define_DINConvAE(NbProjection[comptUpdate],model_AE,x_train.shape,alpha)
                        if flagTrOuputWOMissingData == 1:
                            global_model_FP.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                        else:
                            global_model_FP_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                    elif flagOptimMethod == 1:
                        if (iter > IterTrainAE) & (flagLoadModelAE == 1):
                            print("..... Make trainable AE parameters")
                            for layer in encoder.layers:
                                layer.trainable = True
                            for layer in decoder.layers:
                                layer.trainable = True
                        
                        NBProjCurrent = NbProjection[comptUpdate]
                        NBGradCurrent = NbGradIter[comptUpdate]
                        print("..... Update/initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection[comptUpdate],NbGradIter[comptUpdate]))
                        global_model_Grad,global_model_Grad_Masked = define_GradDINConvAE(NbProjection[comptUpdate],NbGradIter[comptUpdate],model_AE,x_train.shape,gradModel,gradMaskModel,flagGradModel)
                        if flagTrOuputWOMissingData == 1:
                            global_model_Grad.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                        else:
                            global_model_Grad_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                        
                    if comptUpdate < len(NbProjection)-1:
                        comptUpdate += 1
                
                 # gradient descent iteration                    
                if flagOptimMethod == 0:  
                    if flagTrOuputWOMissingData == 1:
                        history = global_model_FP.fit([x_train_init,mask_train],x_train,
                                          batch_size=batch_size,
                                          epochs = NbEpoc,
                                          verbose = 1, 
                                          validation_split=val_split)
                    else:
                        history = global_model_FP_Masked.fit([x_train_init,mask_train],[np.zeros((x_train_init.shape[0],1))],
                                          batch_size=batch_size,
                                          epochs = NbEpoc,
                                          verbose = 1, 
                                          validation_split=val_split)
                elif flagOptimMethod == 1:
                    if flagTrOuputWOMissingData == 1:
                        history = global_model_Grad.fit([x_train_init,mask_train],x_train,
                                          batch_size=batch_size,
                                          epochs = NbEpoc,
                                          verbose = 1, 
                                          validation_split=val_split)
                    else:
                        history = global_model_Grad_Masked.fit([x_train_init,mask_train],[np.zeros((x_train_init.shape[0],1))],
                                          batch_size=batch_size,
                                          epochs = NbEpoc,
                                          verbose = 1, 
                                          validation_split=val_split)
                                        
                if flagOptimMethod == 0:  
                    x_train_pred    = global_model_FP.predict([x_train_init,mask_train])
                    x_test_pred     = global_model_FP.predict([x_test_init,mask_test])
                elif flagOptimMethod == 1:
                    x_train_pred    = global_model_Grad.predict([x_train_init,mask_train])
                    x_test_pred     = global_model_Grad.predict([x_test_init,mask_test])
            
                mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp = eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,
                                           mask_test,x_test,x_test_missing,x_test_pred)
                
                print(".......... iter %d"%(iter))
                print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
                print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
                print('....')
                print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
                print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
                print('....')
                print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
                print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))

                # interpolation and reconstruction score for the center image
                # when dealing with time series
                if x_train_init.shape[3] > 0 :
                    
                    dWCenter    = 32  
                    #indiiCenter = np.arange(dWCenter,x_train_init.shape[1]-dWCenter)
                    #indjjCenter = np.arange(dWCenter,x_train_init.shape[2]-dWCenter)
                    
                    dT = np.floor( x_train_init.shape[3] / 2 ).astype(int)
                    mse_train_center        = np.mean( (x_train_pred[:,:,:,dT] - x_train[:,:,:,dT] )**2 )
                    mse_train_center_interp = np.sum( (x_train_pred[:,:,:,dT]  - x_train[:,:,:,dT] )**2 * (1.-mask_train[:,:,:,dT])  ) / np.sum( (1.-mask_train[:,:,:,dT]) )
                    
                    mse_test_center         = np.mean( (x_test_pred[:,:,:,dT] - x_test[:,:,:,dT] )**2 )
                    mse_test_center_interp  = np.sum( (x_test_pred[:,:,:,dT]  - x_test[:,:,:,dT] )**2 * (1.-mask_test[:,:,:,dT])  ) / np.sum( (1-mask_test[:,:,:,dT]) )
                    
                    var_train_center        = np.var(  x_train[:,:,:,dT] )
                    var_test_center         = np.var(  x_test[:,:,:,dT] )
                    
                    exp_var_train_center         = 1.0 - mse_train_center / var_train_center
                    exp_var_train_interp_center  = 1.0 - mse_train_center_interp / var_train_center
                    exp_var_test_center          = 1.0 - mse_test_center  / var_test_center
                    exp_var_test_interp_center   = 1.0 - mse_test_center_interp/ var_test_center
                print('.... Performance for "center" image')
                print('.... Image center variance (Tr)  : %.2f'%var_train_center)
                print('.... Image center variance (Tt)  : %.2f'%var_test_center)
                print('.... Error for all data (Tr)     : %.2e %.2f%%'%(mse_train_center*stdTr**2,100.*exp_var_train_center))
                print('.... Error for all data (Tt)     : %.2e %.2f%%'%(mse_test_center*stdTr**2,100.*exp_var_test_center))
                print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_center_interp*stdTr**2,100.*exp_var_train_interp_center))
                print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_center_interp*stdTr**2 ,100.*exp_var_test_interp_center))                    
                print('   ')
                
                # AE performance of the trained AE applied to gap-free data
                rec_AE_Tr     = model_AE.predict([x_train,np.ones((mask_train.shape))])
                rec_AE_Tt     = model_AE.predict([x_test,np.ones((mask_test.shape))])
                
                exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
                
                #rec_PCA_Tt       = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
                #rec_PCA_Tt[:,DimAE:] = 0.
                #rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
                #mse_PCA_Tt       = np.mean( (rec_PCA_Tt - np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))**2 )
                #exp_var_PCA_Tt   = 1. - mse_PCA_Tt / var_Tt


                print(".......... Auto-encoder performance when applied to gap-free data")
                print('.... explained variance AE (Tr)  : %.2f%%'%(100.*exp_var_AE_Tr))
                print('.... explained variance AE (Tt)  : %.2f%%'%(100.*exp_var_AE_Tt))
                
                if flagUseMaskinEncoder == 1:
                    rec_AE_Tr     = model_AE.predict([x_train,np.zeros((mask_train.shape))])
                    rec_AE_Tt     = model_AE.predict([x_test,np.zeros((mask_train.shape))])
                
                    exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
                
                    print('.... explained variance AE (Tr) with mask  : %.2f%%'%(100.*exp_var_AE_Tr))
                    print('.... explained variance AE (Tt) with mask  : %.2f%%'%(100.*exp_var_AE_Tt))
                print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1]))
                print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))  

                ## update training data
                if iter > IterUpdateInit:
                    print('')
                    x_train_init = mask_train    * x_train_missing + (1. - mask_train) * x_train_pred
                    x_test_init  = mask_test     * x_test_missing    +  (  1. - mask_test  ) * x_test_pred
                    
                if flagSaveModel == 1:
                    genSuffixModel = '_Alpha%03d'%(100*alpha[0]+10*alpha[1]+alpha[2])
                    if flagUseMaskinEncoder == 1:
                        genSuffixModel = genSuffixModel+'_MaskInEnc'
                        if stdMask  > 0:
                            genSuffixModel = genSuffixModel+'_Std%03d'%(100*stdMask)

                    if flagOptimMethod == 0:  
                        if flagTrOuputWOMissingData == 1:
                            genSuffixModel = genSuffixModel+'_AETRwoMissingData'+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))
                        else:
                            genSuffixModel = genSuffixModel+'_AE'+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))
                    elif flagOptimMethod == 1:  
                        if flagTrOuputWOMissingData == 1:
                            #genSuffixModel = 'AETRwoMissingData'+str('%02d'%(flagAEType))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))
                            genSuffixModel = genSuffixModel+'GradAETRwoMissingData'+str('%02d'%(flagAEType))+str('_%02d'%(flagGradModel))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))+'_Grad'+str('%02d'%(NBGradCurrent))
                        else:
                            genSuffixModel = genSuffixModel+'GradAE'+str('%02d'%(flagAEType))+str('_%02d'%(flagGradModel))+'_D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))+'_Grad'+str('%02d'%(NBGradCurrent))
                    
                        
                    
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_Encoder_iter%03d'%(iter)+'.mod'
                    print('.................. Encoder '+fileMod)
                    encoder.save(fileMod)
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_Decoder_iter%03d'%(iter)+'.mod'
                    print('.................. Decoder '+fileMod)
                    decoder.save(fileMod)
                    if flagOptimMethod == 1:
                        fileMod = dirSAVE+genFilename+genSuffixModel+'_GradModel_iter%03d'%(iter)+'.mod'
                        print('.................. GadModel '+fileMod)
                        gradModel.save(fileMod)
                        fileMod = dirSAVE+genFilename+genSuffixModel+'_GradMaskModel_iter%03d'%(iter)+'.mod'
                        print('.................. Decoder '+fileMod)
                        gradMaskModel.save(fileMod)
                        
                        
                # generate a figure
                flagSaveFig = 1
                if flagSaveFig == 1:
                    np.random.seed(100)
                    indexes_test = np.random.permutation(x_train.shape[0])
        
                    figName = dirSAVE+'FIGS/'+genFilename+genSuffixModel+'_examplesTr_iter%03d'%(iter)+'.pdf'
                
                    idT = int(np.floor(x_train.shape[3]/2))
                    plt.figure()
                    for ii in range(3):
                        Vmin = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.05 )
                        Vmax = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.95 )
                        plt.subplot(4, 3, ii + 1)
                        plt.axis('off')
                        maskFig = np.copy(mask_train[indexes_test[ii],:,:,idT].squeeze())
                        maskFig[ maskFig == 0] = np.float('NaN')
                        plt.imshow(x_train[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                        plt.title('%d\n' %(indexes_test[ii]))
                        plt.subplot(4, 3, ii + 1+3)
                        plt.axis('off')
                        plt.imshow(x_train_missing[indexes_test[ii],:,:,idT].squeeze() * maskFig, cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                        plt.subplot(4, 3, ii + 1+6)
                        plt.axis('off')
                        plt.imshow(x_train_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                        plt.subplot(4, 3, ii + 1+9)
                        plt.axis('off')
                        plt.imshow(rec_AE_Tr[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                    print('..... Save figure: '+figName)
                    plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
                                orientation='portrait', quality=98)

                    indexes_test = np.random.permutation(x_test.shape[0])
        
                    figName = dirSAVE+'FIGS/'+genFilename+genSuffixModel+'_examplesTt_iter%03d'%(iter)+'.pdf'
                
                    idT = int(np.floor(x_test.shape[3]/2))
                    plt.figure()
                    for ii in range(3):
                        Vmin = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.05 )
                        Vmax = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.95 )
                        maskFig = np.copy(mask_test[indexes_test[ii],:,:,idT].squeeze())
                        maskFig[ maskFig == 0] = np.float('NaN')
                        plt.subplot(4, 3, ii + 1)
                        plt.axis('off')
                        plt.imshow(x_test[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                        plt.title('%d\n GT' %(indexes_test[ii]))
                        plt.subplot(4, 3, ii + 1+3)
                        plt.axis('off')
                        plt.imshow(x_test_missing[indexes_test[ii],:,:,idT].squeeze() * maskFig, cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                        plt.subplot(4, 3, ii + 1+6)
                        plt.axis('off')
                        plt.imshow(x_test_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                        plt.subplot(4, 3, ii + 1+9)
                        plt.axis('off')
                        plt.imshow(rec_AE_Tt[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                    print('..... Save figure: '+figName)
                    plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
                                orientation='portrait', quality=98)
            
        ###############################################################
        ## train ZeroPaddingConv-AE
        elif flagProcess[kk] == 5:        

            # masked AE model
            x_input         = keras.layers.Input((x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            mask            = keras.layers.Input((x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            
            x_proj = model_AE([x_input,mask])
            x_proj = keras.layers.Multiply()([x_proj,mask])
            
            model_AE_Masked  = keras.models.Model([x_input,mask],[x_proj])
            model_AE_Masked.summary()
            
           # PCA decomposition for comparison
            pca              = decomposition.PCA(DimCAE)
            pca.fit(np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])))
            
            rec_PCA_Tt       = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
            rec_PCA_Tt[:,DimCAE:] = 0.
            rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
            mse_PCA_Tt       = np.mean( (rec_PCA_Tt - x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))**2 )
            var_Tt           = np.mean( (x_test-np.mean(x_train,axis=0))** 2 )
            exp_var_PCA_Tt   = 1. - mse_PCA_Tt / var_Tt
            
            print(".......... PCA Dim = %d"%(DimCAE))
            print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1]))
            print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))

            
            # model compilation
            # model fit
            Niter      = 100
            IterUpdate     = [0,3,6,9,12,15]
            lrUpdate       = [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
            flagLoadModelAE = 0
            fileAEModelInit = './MNIST/mnist_DINConvAE_AE02N30W02_Nproj15_Decoder_iter018.mod'#'./MNIST/mnist_DINConvAE_AE02N30W02_Nproj10_Encoder_iter011.mod'
            fileAEModelInit = './MNIST/mnist_DINConvAE_AE03N30W02_Nproj15_Encoder_iter018.mod'
            iterInit        = 0
            
            ## initialization
            x_train_init = np.copy(x_train_missing)
            x_test_init  = np.copy(x_test_missing)

            comptUpdate = 0
            if flagLoadModelAE == 1:
                print('.................. Load Encoder/Decoder '+fileAEModelInit)
                encoder.load_weights(fileAEModelInit)
                decoder.load_weights(fileAEModelInit.replace('Encoder','Decoder'))

                comptUpdate   = 4
                NBProjCurrent = NbProjection[comptUpdate-1]
                print("..... Initialize number of projections in DINCOnvAE model # %d"%(NbProjection[comptUpdate-1]))
                if flagTrOuputWOMissingData == 1:
                    model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                else:
                    model_AE_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
            else:
                iterInit        = 0
                                
                
            for iter in range(iterInit,Niter):
                if iter == IterUpdate[comptUpdate]:
                    # update DINConvAE model
                    print("..... Update learning rate # %f"%(lrUpdate[comptUpdate]))
                    if flagTrOuputWOMissingData == 1:
                        model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                    else:
                        model_AE_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                    
                    if comptUpdate < len(IterUpdate)-1:
                        comptUpdate += 1
                  
                # gradient descent iteration
                if flagTrOuputWOMissingData == 1:
                    history = model_AE.fit(x_train_missing,x_train,
                                      batch_size=batch_size,
                                      epochs = NbEpoc,
                                      verbose = 1, 
                                      validation_split=0.1)
                    
                else:
                    history = model_AE_Masked.fit([x_train_missing,mask_train],[x_train_missing],
                                      batch_size=batch_size,
                                      epochs = NbEpoc,
                                      verbose = 1, 
                                      validation_split=0.1)
        
                x_train_pred    = model_AE.predict([x_train_missing,mask_train])
                x_test_pred     = model_AE.predict([x_test_missing,mask_test])
            
                mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp = eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,
                                           mask_test,x_test,x_test_missing,x_test_pred)
                
                if flagTrOuputWOMissingData == 1:
                    print(".......... Training wo missing data")
                print(".......... iter %d"%(iter))
                print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
                print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
                print('....')
                print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
                print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
                print('....')
                print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
                print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))
                
                # AE performance of the trained AE applied to gap-free data
                rec_AE_Tr     = model_AE.predict([x_train,np.ones((x_train.shape))])
                rec_AE_Tt     = model_AE.predict([x_test,np.ones((x_test.shape))])
                
                
                exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
                
                #rec_PCA_Tt       = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
                #rec_PCA_Tt[:,DimCAE:] = 0.
                #rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
                #mse_PCA_Tt       = np.mean( (rec_PCA_Tt - np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))**2 )
                #exp_var_PCA_Tt   = 1. - mse_PCA_Tt / var_Tt

                print(".......... Auto-encoder performance when applied to gap-free data")
                print('.... explained variance AE (Tr)  : %.2f%%'%(100.*exp_var_AE_Tr))
                print('.... explained variance AE (Tt)  : %.2f%%'%(100.*exp_var_AE_Tt))
                print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1]))
                print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))  
                
                if flagSaveModel == 1:
                    genSuffixModel = 'ZeroAE'+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))
                    if flagTrOuputWOMissingData == 1:
                        genSuffixModel = genSuffixModel.replace('ZeroAE','ZeroAE_TRwoMissingData')
                    
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_Encoder_iter%03d'%(iter)+'.mod'
                    print('.................. Encoder '+fileMod)
                    encoder.save(fileMod)
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_Decoder_iter%03d'%(iter)+'.mod'
                    print('.................. Decoder '+fileMod)
                    decoder.save(fileMod)

      ###############################################################
        ## train AE wo missing data
        elif flagProcess[kk] == 6:        

            # masked AE model
            x_input         = keras.layers.Input((x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            mask            = keras.layers.Input((x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            
            x_proj = model_AE([x_input,mask])
            x_proj = keras.layers.Multiply()([x_proj,mask])
            
            model_AE_Masked  = keras.models.Model([x_input,mask],[x_proj])
            model_AE_Masked.summary()
            
           # PCA decomposition for comparison
            pca              = decomposition.PCA(DimCAE)
            pca.fit(np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])))
            
            rec_PCA_Tt       = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
            rec_PCA_Tt[:,DimCAE:] = 0.
            rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
            rec_PCA_Tt       = np.reshape(rec_PCA_Tt,(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

            rec_PCA_Tr       = pca.transform(np.reshape(x_train,(x_train.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
            rec_PCA_Tr[:,DimCAE:] = 0.
            rec_PCA_Tr       = pca.inverse_transform(rec_PCA_Tr)
            rec_PCA_Tr       = np.reshape(rec_PCA_Tr,(x_train.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

            exp_var_PCA_Tr,exp_var_PCA_Tt = eval_AEPerformance(x_train,rec_PCA_Tr,x_test,rec_PCA_Tt)

            mse_PCA_Tt       = np.mean( (rec_PCA_Tt - x_test)**2 )
            var_Tt           = np.mean( (x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))-pca.mean_)** 2 )
            exp_var_PCA_Tt   = 1. - mse_PCA_Tt / var_Tt
            
            print(".......... PCA Dim = %d"%(DimAE))
            print('.... explained variance PCA (Tr) : %.2f%% %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1],100.*exp_var_PCA_Tr))
            print('.... explained variance PCA (Tt) : %.2f%% %.2f%%'%(100.*exp_var_PCA_Tt,100.*(1. - mse_PCA_Tt / var_Tt)))

            print("..... Regularization parameters: dropout = %.3f, wl2 = %.2E"%(dropout,wl2))
           
            # model compilation
            # model fit
            Niter      = 100
            IterUpdate     = [0,3,6,12,20,24]
            lrUpdate       = [1e-3,1e-4,1e-5,1e-6,1e-5,1e-5]
            comptUpdate = 0
            
            for iter in range(0,Niter):
                if iter == IterUpdate[comptUpdate]:
                    # update DINConvAE model
                    print("..... Update learning rate # %f"%(lrUpdate[comptUpdate]))
                    model_AE_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                    
                    if comptUpdate < len(IterUpdate)-1:
                        comptUpdate += 1
                  
                # gradient descent iteration
                history = model_AE.fit([x_train,np.ones((x_train.shape))],[x_train],
                                  batch_size=batch_size,
                                  epochs = NbEpoc,
                                  verbose = 1, 
                                  validation_split=0.1)
                        
                # AE performance of the trained AE applied to gap-free data
                rec_AE_Tr     = model_AE.predict([x_train,np.ones((x_train.shape))])
                rec_AE_Tt     = model_AE.predict([x_test,np.ones((x_test.shape))])
                                
                exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
                
                print(".......... Auto-encoder performance when applied to gap-free data")
                print('.... explained variance AE (Tr)  : %.2f%%'%(100.*exp_var_AE_Tr))
                print('.... explained variance AE (Tt)  : %.2f%%'%(100.*exp_var_AE_Tt))
                print('.... explained variance PCA (Tr) : %.2f%%'%(100.*exp_var_PCA_Tr))
                print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))  
                
                if flagSaveModel == 1:
                    genSuffixModel = 'REFwoMissingData'+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))
    
                    
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_Encoder_iter%03d'%(iter)+'.mod'
                    print('.................. Encoder '+fileMod)
                    encoder.save(fileMod)
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_Decoder_iter%03d'%(iter)+'.mod'
                    print('.................. Decoder '+fileMod)
                    decoder.save(fileMod)

        ###############################################################
        ## Performance evaluation for missing data usng PCA
        elif flagProcess[kk] == 7:        
            NbProjection     = 15   
            
            print('.................. Configuration ')
            print('....   DimAE = %d '%(DimAE))
            print('....   Nsq   = %d '%(Nsquare))
            print('....   Wsq   = %d '%(Wsquare))
            print('....   Dataset    = %d '%(flagDataset))
            print('....   AEType     = %d '%(flagAEType))
            print('....   NbProj     = %d '%(NbProjection))

           # PCA decomposition for comparison
            pca              = decomposition.PCA(DimAE)
            pca.fit(np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])))
            
            # train and test a classifier from the learned feature space
            classifier = defineClassifier(DimAE,num_classes)

            classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            classifier.summary()
        
            z_train = keras.utils.to_categorical(y_train, num_classes)
            z_test  = keras.utils.to_categorical(y_test, num_classes)
            
            # apply model
            rec_PCA_Tr = np.copy(x_train_missing)
            rec_PCA_Tt = np.copy(x_test_missing)
            
            for nn in range(0,NbProjection):
                rec_PCA_Tr       = pca.transform(np.reshape(rec_PCA_Tr,(rec_PCA_Tr.shape[0],rec_PCA_Tr.shape[1]*rec_PCA_Tr.shape[2]*rec_PCA_Tr.shape[3])))
                rec_PCA_Tr[:,DimAE:] = 0.
                rec_PCA_Tr       = pca.inverse_transform(rec_PCA_Tr)
                rec_PCA_Tr       = np.reshape(rec_PCA_Tr,(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
                rec_PCA_Tr       = rec_PCA_Tr * mask_train.astype(float) + x_train_missing  * (1.0-mask_train)
            
                rec_PCA_Tt       = pca.transform(np.reshape(rec_PCA_Tt,(rec_PCA_Tt.shape[0],rec_PCA_Tt.shape[1]*rec_PCA_Tt.shape[2]*rec_PCA_Tt.shape[3])))
                rec_PCA_Tt[:,DimAE:] = 0.
                rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
                rec_PCA_Tt       = np.reshape(rec_PCA_Tt,(x_test.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))        
                rec_PCA_Tt       = rec_PCA_Tt  * mask_test.astype(float) + x_test_missing * (1.0-mask_test)
            
            rec_PCA_Tr       = pca.transform(np.reshape(rec_PCA_Tr,(rec_PCA_Tr.shape[0],rec_PCA_Tr.shape[1]*rec_PCA_Tr.shape[2]*rec_PCA_Tr.shape[3])))
            rec_PCA_Tr[:,DimAE:] = 0.
            rec_PCA_Tr       = pca.inverse_transform(rec_PCA_Tr)
            
            rec_PCA_Tt       = pca.transform(np.reshape(rec_PCA_Tt,(rec_PCA_Tt.shape[0],rec_PCA_Tt.shape[1]*rec_PCA_Tt.shape[2]*rec_PCA_Tt.shape[3])))
            rec_PCA_Tt[:,DimAE:] = 0.
            rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
            
            x_train_pred    = np.reshape(rec_PCA_Tr,(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            x_test_pred     = np.reshape(rec_PCA_Tt,(x_test.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))        


            # AE performance wo missing data
            rec_PCA_Tt       = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
            rec_PCA_Tt[:,DimAE:] = 0.
            rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
            mse_PCA_Tt       = np.mean( (rec_PCA_Tt - x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))**2 )
            var_Tt           = np.mean( (x_test-np.mean(x_train,axis=0))** 2 )
            exp_var_PCA_Tt   = 1. - mse_PCA_Tt / var_Tt
            
            rec_PCA_Tr       = np.reshape(rec_PCA_Tr,(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            rec_PCA_Tt       = np.reshape(rec_PCA_Tt,(x_test.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            
            print(".......... PCA Dim = %d"%(DimAE))
            print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimAE-1]))
            print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))

            #### Performance Summary
            mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp = eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,
                                       mask_test,x_test,x_test_missing,x_test_pred)
            
            # classification performance
            feat_train       = pca.transform(np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])))
            feat_train       = feat_train[:,0:DimAE]
            feat_test        = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_train.shape[3])))
            feat_test        = feat_test[:,0:DimAE]

            batch_size_classif = 128
            epochs_classif     = 20
            classifier.fit(feat_train, z_train, batch_size=batch_size_classif, epochs=epochs_classif, verbose=1,
                            validation_data=(feat_test, z_test))

            
                
            score = classifier.evaluate(feat_test, z_test, verbose=0)

            print(".......... Reconstruction performance with Missing Data")
            print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
            print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
            print('....')
            print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
            print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
            print('....')
            print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
            print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))


            print(".......... Classification performance")
            #print('Test loss     : %.2f', 100*score[0])
            print('.... Test accuracy : %.2f%%'%(100*score[1]))
            
            # generate a figure
            np.random.seed(100)
            indexes_test = np.random.permutation(x_test.shape[0])

            
            figName = 'MNIST/FIGS/resDINEOF'+str('%02d'%(DimAE))+'_N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'.pdf'
            
            # generate a figure
            np.random.seed(100)
            indexes_test = np.random.permutation(x_train.shape[0])

            figName = dirSAVE + 'FIGS/' + genFilename + '_explesTt.pdf'
        
            idT = int(np.floor(x_train.shape[3]/2))
            plt.figure()
            for ii in range(3):
                Vmin = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.05 )
                Vmax = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.95 )
                plt.subplot(4, 3, ii + 1)
                plt.axis('off')
                maskFig = np.copy(mask_train[indexes_test[ii],:,:,idT].squeeze())
                maskFig[ maskFig[:] == 0] = np.float('NaN')
                plt.imshow(x_train[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.title('%d\n' %(indexes_test[ii]))
                plt.subplot(4, 3, ii + 1+3)
                plt.axis('off')
                plt.imshow(x_train_missing[indexes_test[ii],:,:,idT].squeeze() * maskFig, cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(4, 3, ii + 1+6)
                plt.axis('off')
                plt.imshow(x_train_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(4, 3, ii + 1+9)
                plt.axis('off')
                plt.imshow(rec_PCA_Tr[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
            print('..... Save figure: '+figName)
            plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', quality=98)

            indexes_test = np.random.permutation(x_test.shape[0])
        
            #figName = dirSAVE+'FIGS/'+genFilename+genSuffixModel+'_examplesTt_iter%03d'%(iter)+'.pdf'
            figName = dirSAVE + 'FIGS/' + genFilename + '_explesTt.pdf'
            
            idT = int(np.floor(x_test.shape[3]/2))
            plt.figure()
            for ii in range(3):
                Vmin = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.05 )
                Vmax = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.95 )
                maskFig = np.copy(mask_test[indexes_test[ii],:,:,idT].squeeze())
                maskFig[ maskFig[:] == 0] = np.float('NaN')
                plt.subplot(5, 3, ii + 1)
                plt.axis('off')
                plt.imshow(x_test[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.title('%d\n GT' %(indexes_test[ii]))
                plt.subplot(5, 3, ii + 1+3)
                plt.axis('off')
                plt.imshow(x_test_missing[indexes_test[ii],:,:,idT].squeeze() * maskFig, cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, 3, ii + 1+6)
                plt.axis('off')
                plt.imshow(x_test_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, 3, ii + 1+9)
                plt.axis('off')
                plt.imshow(rec_PCA_Tt[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
            print('..... Save figure: '+figName)
            plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', quality=98)

       ###############################################################
        ## Performance evaluation for Zero-DinAE and AE  wo missing data
        elif flagProcess[kk] == 8:        

            # masked AE model
            x_input         = keras.layers.Input((x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            mask            = keras.layers.Input((x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            
            x_proj = model_AE(x_input)
            x_proj = keras.layers.Multiply()([x_proj,mask])
            
            model_AE_Masked  = keras.models.Model([x_input,mask],[x_proj])
            model_AE_Masked.summary()
            
            # model compilation
            # model fit
            fileAEModel = './MNIST/mnist_DINConvAE_ZeroAE02N03W04_Encoder_iter009.mod'#'./MNIST/mnist_DINConvAE_AE02N30W02_Nproj10_Encoder_iter011.mod'
            fileAEModel = './MNIST/mnist_DINConvAE_ZeroAE02N03W04_Encoder_iter009.mod'#'./MNIST/mnist_DINConvAE_AE02N30W02_Nproj10_Encoder_iter011.mod'
            fileAEModel = './MNIST/mnist_DINConvAE_REFwoMissingData02D20_Encoder_iter020.mod'
            fileAEModel = './MNIST/mnist_DINConvAE_ZeroAE_TRwoMissingData02N03W04_Encoder_iter020.mod'
            fileAEModel = './MNIST/fashion_mnist_DINConvAE_REFwoMissingData02D20_Encoder_iter005.mod'
            fileAEModel = './MNIST/fashion_mnist_DINConvAE_ZeroAE_TRwoMissingData02N03W04_Encoder_iter020.mod'
            fileAEModel = './MNIST/fashion_mnist_DINConvAE_ZeroAE02D20N20W02_Encoder_iter010.mod'
            #fileAEModel = './MNIST/fashion_mnist_DINConvAE_ZeroAE_TRwoMissingData02D20N03W04_Encoder_iter010.mod'
            fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/sstNATL60_METOP_DINConvAE_ZeroAE_TRwoMissingData02D20N00W00_Encoder_iter099.mod'
            fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/sstNATL60_METOP_DINConvAE_ZeroAE02D20N00W00_Encoder_iter099.mod'
            
            #fileAEModel = fileAEModel.replace('N03','N%02d'%(Nsquare))
            #fileAEModel = fileAEModel.replace('W04','W%02d'%(Wsquare))

            print('.................. Configuration ')
            print('....   DimAE = %d '%(DimAE))
            print('....   Nsq   = %d '%(Nsquare))
            print('....   Wsq   = %d '%(Wsquare))
            print('....   Dataset   = %d '%(flagDataset))
            print('....   AEType    = %d '%(flagAEType))

            print('.................. Load Encoder/Decoder '+fileAEModel)
            encoder.load_weights(fileAEModel)
            decoder.load_weights(fileAEModel.replace('Encoder','Decoder'))
                               
            # apply model                        
            x_train_pred    = model_AE.predict(x_train_missing)
            x_test_pred     = model_AE.predict(x_test_missing)        

            # train and test a classifier from the learned feature space
            classifier = defineClassifier(DimAE,num_classes)

            classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            classifier.summary()
        
            z_train = keras.utils.to_categorical(y_train, num_classes)
            z_test  = keras.utils.to_categorical(y_test, num_classes)
            
            if 1*1: # Conv-AE
              feat_train = encoder.predict(x_train).reshape((x_train.shape[0],DimAE))
              feat_test  = encoder.predict(x_test).reshape((x_test.shape[0],DimAE))
            else: # PCA
              feat_train = pca.transform(np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2])))
              feat_test  = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2])))

            batch_size_classif = 128
            epochs_classif     = 20
            classifier.fit(feat_train, z_train, batch_size=batch_size_classif, epochs=epochs_classif, verbose=1,
                            validation_data=(feat_test, z_test))


            mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp = eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,
                                       mask_test,x_test,x_test_missing,x_test_pred)
            
            
            #### Performance Summary
            print(".......... Reconstruction performance with Missing Data")
            print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
            print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
            print('....')
            print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
            print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
            print('....')
            print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
            print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))

                
            # AE performance of the trained AE applied to gap-free data
            rec_AE_Tr     = model_AE.predict(x_train)
            rec_AE_Tt     = model_AE.predict(x_test)
                        
            exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
            
            print(".......... Auto-encoder performance when applied to gap-free data")
            print('.... explained variance AE (Tr)  : %.2f%%'%(100.*exp_var_AE_Tr))
            print('.... explained variance AE (Tt)  : %.2f%%'%(100.*exp_var_AE_Tt))
              
      
            score = classifier.evaluate(feat_test, z_test, verbose=0)
            print(".......... Classification performance")
            #print('Test loss     : %.2f', 100*score[0])
            print('.... Test accuracy : %.2f%%'%(100*score[1]))
            

            # generate a figure
            np.random.seed(100)
            indexes_test = np.random.permutation(x_test.shape[0])

            figName = fileAEModel.replace('.mod','_examples.pdf')
            figName = figName.replace('MNIST/','MNIST/FIGS/')
            SuffName = '_N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))
            figName = figName.replace('.pdf',SuffName+'.pdf')
            
            idT = int(np.floor(x_test.shape[3]/2))
            plt.figure()
            for ii in range(5):
                plt.subplot(4, 5, ii + 1)
                plt.axis('off')
                plt.imshow(x_test[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
                plt.title('%d\n GT:%i' %(indexes_test[ii],y_test[indexes_test[ii]]))
                plt.subplot(4, 5, ii + 1+5)
                plt.axis('off')
                plt.imshow(x_test_missing[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
                plt.subplot(4, 5, ii + 1+10)
                plt.axis('off')
                plt.imshow(x_test_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
                plt.subplot(4, 5, ii + 1+15)
                plt.axis('off')
                plt.imshow(rec_AE_Tt[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
#            plt.figure()
#            for ii in range(5):
#                plt.subplot(4, 5, ii + 1)
#                plt.axis('off')
#                plt.imshow(x_test[indexes_test[ii],:,:], cmap=plt.cm.gray_r)
#                plt.title('%d\n GT:%i' %(indexes_test[ii],y_test[indexes_test[ii]]))
#                plt.subplot(4, 5, ii + 1+5)
#                plt.axis('off')
#                plt.imshow(x_test_missing[indexes_test[ii],:,:], cmap=plt.cm.gray_r)
#                plt.subplot(4, 5, ii + 1+10)
#                plt.axis('off')
#                plt.imshow(x_test_pred[indexes_test[ii],:,:], cmap=plt.cm.gray_r)
#                plt.subplot(4, 5, ii + 1+15)
#                plt.axis('off')
#                plt.imshow(rec_AE_Tt[indexes_test[ii],:,:], cmap=plt.cm.gray_r)
#            print('..... Save figure: '+figName)
            plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', quality=98)

        ###############################################################
        ## Performance evaluation for FP-ConvAE and G-ConvAE 
        elif flagProcess[kk] == 9:        
            NbProjection    = 0   
            NbGradIter      = 0
            flagOptimMethod = 0            
            flagEvalClassif = 0

            # model compilation
            # model fit
            fileAEModel = './MNIST/mnist_DINConvAE_ZeroAE02N03W04_Encoder_iter009.mod'#'./MNIST/mnist_DINConvAE_AE02N30W02_Nproj10_Encoder_iter011.mod'
            fileAEModel = './MNIST/mnist_DINConvAE_AE02N06W04_Nproj15_Encoder_iter018.mod'#'./MNIST/mnist_DINConvAE_AE02N30W02_Nproj10_Encoder_iter011.mod'
            #fileAEModel = './MNIST/mnist_DINConvAE_REFwoMissingData02D20_Encoder_iter020.mod'
            fileAEModel = './MNIST/mnist_DINConvAE_AETRwoMissingData02N06W04_Nproj10_Encoder_iter015.mod'
            fileAEModel = './MNIST/fashion_mnist_DINConvAE_AE02N06W04_Nproj15_Encoder_iter020.mod'
            fileAEModel = './MNIST/fashion_mnist_DINConvAE_AE02D20N06W04_Nproj15_Encoder_iter019.mod'
            
            #fileAEModel = './MNIST/fashion_mnist_DINConvAE_AETRwoMissingData02N06W04_Nproj15_Encoder_iter020.mod'
            #fileAEModel = './MNIST/fashion_mnist_DINConvAE_AE02N06W04_Nproj10_Encoder_iter010.mod'
            #fileAEModel = './MNIST/fashion_mnist_DINConvAE_ZeroAE02N06W04_Encoder_iter009.mod'
            fileAEModel = './MNIST/mnist_DINConvAE_GradAE02_00_D20N30W02_Nproj05_Grad10_Encoder_iter010.mod'
            fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/sstNATL60_METOP_DINConvAE_AETRwoMissingData02D20N00W00_Nproj10_Encoder_iter014.mod'
            fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/sstNATL60_METOP_DINConvAE_AE02D20N00W00_Nproj10_Encoder_iter014.mod'
            
            #fileAEModel = './MNIST/mnist_DINConvAE_v2_REFwoMissingData02D20_Encoder_iter001.mod'
            fileAEModel = './MNIST/mnist_DINConvAE_v3__Alpha100_AE02D20N20W02_Nproj00_Encoder_iter004.mod'
            
            if 1*0 :
                fileAEModel = './MNIST/mnist_DINConvAE_v3__Alpha100_AE02D20N20W02_Nproj15_Encoder_iter015.mod'
                #fileAEModel = './MNIST/mnist_DINConvAE_v2__Alpha100GradAE02_00_D20N20W02_Nproj00_Grad14_Encoder_iter015.mod'
                #fileAEModel = './MNIST/mnist_DINConvAE_v2__Alpha100GradAE02_00_D20N03W04_Nproj00_Grad10_Encoder_iter010.mod'
                #fileAEModel = './MNIST/mnist_DINConvAE_v2__Alpha101_AE02D20N20W02_Nproj15_Encoder_iter026.mod'
                
            fileAEModel = fileAEModel.replace('N20','N%02d'%(Nsquare))
            fileAEModel = fileAEModel.replace('W02','W%02d'%(Wsquare))
            
            
            if 1*0:
                #fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SST_128_512_005_dW000WFilter005_NFilter040_RU003_LR004_GradAE07_00_D20N00W00_Nproj02_Grad03_Encoder_iter015.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SST_128_512_005GradAE06_00_D20N00W00_Nproj00_Grad04_Encoder_iter015.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SST_128_512_005GradAE08_00_D20N00W00_Nproj02_Grad03_Encoder_iter015.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SST_128_512_005AE08D20N00W00_Nproj05_Encoder_iter015.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SST_128_512_005_dW000WFilter005_NFilter040_RU003_LR004_AE07D20N00W00_Nproj05_Encoder_iter015.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SST_128_512_005AE06D20N00W00_Nproj05_Encoder_iter010.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SST_128_512_005GradAE08_00_D20N00W00_Nproj00_Grad04_Encoder_iter015.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SST_128_512_005AE06D20N00W00_Nproj05_Encoder_iter015.mod'
                
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SSTAnomaly_128_512_005_dW000WFilter005_NFilter040_RU003_LR004_AE07D20N00W00_Nproj05_Encoder_iter015.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_dW000WFilter005_NFilter040_RU003_LR004_Alpha100_AE07D20N00W00_Nproj05_Encoder_iter017.mod'
                
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_Alpha100_AE02D20N00W00_Nproj00_Encoder_iter009.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_dW000WFilter009_NFilter020_RU003_LR004woSRREFwoMissingData07D20_Encoder_iter015.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_Alpha100_AE06D20N00W00_Nproj10_Encoder_iter034.mod'
                #fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SSTAnomaly_128_512_005AE08D20N00W00_Nproj05_Encoder_iter015.mod'
        
                #fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SST_128_512_005GradAE06_00_D20N00W00_Nproj00_Grad04_Encoder_iter015.mod'            
                #fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes/model_patchDataset_NATL60withMETOP_SST_128_512_005_dW000WFilter011_NFilter100_RU010_LR004_ZeroAE07D50N00W00_Encoder_iter099.mod'
                fileAEModel = 'Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/model_patchDataset_NATL60withMETOP_SST_128_512_011_dW000WFilter011_NFilter040_RU010_LR004woSR_Alpha100_AE07D40N00W00_Nproj15_Encoder_iter039.mod'
            

            print('.................. Configuration ')
            print('....   DimAE = %d '%(DimAE))
            print('....   Nsq   = %d '%(Nsquare))
            print('....   Wsq   = %d '%(Wsquare))
            print('....   Dataset    = %d '%(flagDataset))
            print('....   AEType     = %d '%(flagAEType))
            print('....   NbProj     = %d '%(NbProjection))
            if flagOptimMethod == 1:        
                print('....   NbGradIter = %d '%(NbGradIter))
            
            print('.................. Load Encoder/Decoder '+fileAEModel)
            encoder.load_weights(fileAEModel)
            decoder.load_weights(fileAEModel.replace('Encoder','Decoder'))
            if flagOptimMethod == 0:
                print("..... Initialize DINCOnvAE model # %d"%(NbProjection))
                global_model_FP,global_model_FP_Masked = define_DINConvAE(NbProjection,model_AE,x_train.shape,alpha)
            elif flagOptimMethod == 1:                
                gradModel,gradMaskModel =  define_GradModel(model_AE,x_train.shape,flagGradModel)
                gradMaskModel.load_weights(fileAEModel.replace('Encoder','GradMaskModel'))
                gradModel.load_weights(fileAEModel.replace('Encoder','GradModel'))
                
                print("..... Initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection,NbGradIter))
                global_model_Grad,global_model_Grad_Masked = define_GradDINConvAE(NbProjection,NbGradIter,model_AE,x_train.shape,gradModel,gradMaskModel,flagGradModel)
            
            # apply model                        
            print("..... Apply model ")
            if flagOptimMethod == 0:  
                x_train_pred    = global_model_FP.predict([x_train_missing,mask_train])
                x_test_pred     = global_model_FP.predict([x_test_missing,mask_test])
            elif flagOptimMethod == 1:
                x_train_pred    = global_model_Grad.predict([x_train_missing,mask_train])
                x_test_pred     = global_model_Grad.predict([x_test_missing,mask_test])

            # train and test a classifier from the learned feature space
            if flagEvalClassif == 1:
                classifier = defineClassifier(DimAE,num_classes)
    
                classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
                classifier.summary()
            
                z_train = keras.utils.to_categorical(y_train, num_classes)
                z_test  = keras.utils.to_categorical(y_test, num_classes)
                
                feat_train = encoder.predict([x_train,mask_train]).reshape((x_train.shape[0],DimAE))
                feat_test  = encoder.predict([x_test,mask_test]).reshape((x_test.shape[0],DimAE))
    
                batch_size_classif = 128
                epochs_classif     = 20
                classifier.fit(feat_train, z_train, batch_size=batch_size_classif, epochs=epochs_classif, verbose=1,
                                validation_data=(feat_test, z_test))


            mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp = eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,
                                       mask_test,x_test,x_test_missing,x_test_pred)
            
            
            #### Performance Summary
            print(".......... Reconstruction performance with Missing Data")
            print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
            print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
            print('....')
            print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
            print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
            print('....')
            print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
            print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))
                
            # interpolation and reconstruction score for the center image
            # when dealing with time series
            if x_train_missing.shape[3] > 0 :
                
                dWCenter    = 32  
                #indiiCenter = np.arange(dWCenter,x_train_init.shape[1]-dWCenter)
                #indjjCenter = np.arange(dWCenter,x_train_init.shape[2]-dWCenter)
                
                dT = np.floor( x_train_missing.shape[3] / 2 ).astype(int)
                mse_train_center        = np.mean( (x_train_pred[:,:,:,dT] - x_train[:,:,:,dT] )**2 )
                mse_train_center_interp = np.sum( (x_train_pred[:,:,:,dT]  - x_train[:,:,:,dT] )**2 * (1.-mask_train[:,:,:,dT])  ) / np.sum( (1.-mask_train[:,:,:,dT]) )
                
                mse_test_center         = np.mean( (x_test_pred[:,:,:,dT] - x_test[:,:,:,dT] )**2 )
                mse_test_center_interp  = np.sum( (x_test_pred[:,:,:,dT]  - x_test[:,:,:,dT] )**2 * (1.-mask_test[:,:,:,dT])  ) / np.sum( (1-mask_test[:,:,:,dT]) )
                
                var_train_center        = np.var(  x_train[:,:,:,dT] )
                var_test_center         = np.var(  x_test[:,:,:,dT] )
                
                exp_var_train_center         = 1.0 - mse_train_center / var_train_center
                exp_var_train_interp_center  = 1.0 - mse_train_center_interp / var_train_center
                exp_var_test_center          = 1.0 - mse_test_center  / var_test_center
                exp_var_test_interp_center   = 1.0 - mse_test_center_interp/ var_test_center
                print('.... Performance for "center" image')
                print('.... Image center variance (Tr)  : %.2f'%var_train_center)
                print('.... Image center variance (Tt)  : %.2f'%var_test_center)
                print('.... Error for all data (Tr)     : %.2e %.2f%%'%(mse_train_center*stdTr**2,100.*exp_var_train_center))
                print('.... Error for all data (Tt)     : %.2e %.2f%%'%(mse_test_center*stdTr**2,100.*exp_var_test_center))
                print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_center_interp*stdTr**2,100.*exp_var_train_interp_center))
                print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_center_interp*stdTr**2 ,100.*exp_var_test_interp_center))                    
                print('   ')

                if flagloadOIData == 1:
                    dWCenter    = 32  
                    #indiiCenter = np.arange(dWCenter,x_train_init.shape[1]-dWCenter)
                    #indjjCenter = np.arange(dWCenter,x_train_init.shape[2]-dWCenter)
                    
                    mse_train_center_OI        = np.mean( (x_train_OI[:,:,:] - x_train[:,:,:,dT] )**2 )
                    mse_train_center_interp_OI = np.sum( (x_train_OI[:,:,:]  - x_train[:,:,:,dT] )**2 * (1.-mask_train[:,:,:,dT])  ) / np.sum( (1.-mask_train[:,:,:,dT]) )
                    
                    mse_test_center_OI         = np.mean( (x_test_OI[:,:,:] - x_test[:,:,:,dT] )**2 )
                    mse_test_center_interp_OI  = np.sum( (x_test_OI[:,:,:]  - x_test[:,:,:,dT] )**2 * (1.-mask_test[:,:,:,dT])  ) / np.sum( (1-mask_test[:,:,:,dT]) )
                                    
                    exp_var_train_center_OI         = 1.0 - mse_train_center_OI / var_train_center
                    exp_var_train_interp_center_OI  = 1.0 - mse_train_center_interp_OI / var_train_center
                    exp_var_test_center_OI          = 1.0 - mse_test_center_OI  / var_test_center
                    exp_var_test_interp_center_OI   = 1.0 - mse_test_center_interp_OI / var_test_center
            
                    print('.... OI: ' + SuffixOI )
                    print('.... OI: Error for all data (Tr)     : %.2e %.2f%%'%(mse_train_center_OI*stdTr**2,100.*exp_var_train_center_OI))
                    print('.... OI: Error for all data (Tt)     : %.2e %.2f%%'%(mse_test_center_OI*stdTr**2,100.*exp_var_test_center_OI))
                    print('.... OI: Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_center_interp_OI*stdTr**2,100.*exp_var_train_interp_center_OI))
                    print('.... OI: Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_center_interp_OI*stdTr**2 ,100.*exp_var_test_interp_center_OI))                    
                    print('   ')

            # AE performance of the trained AE applied to gap-free data
            rec_AE_Tr     = model_AE.predict([x_train,np.ones((mask_train.shape))])
            rec_AE_Tt     = model_AE.predict([x_test,np.ones((mask_test.shape))])
                        
            exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
            
            print(".......... Auto-encoder performance when applied to gap-free data")
            print('.... explained variance AE (Tr)  : %.2f%%'%(100.*exp_var_AE_Tr))
            print('.... explained variance AE (Tt)  : %.2f%%'%(100.*exp_var_AE_Tt))
              
      
            if flagEvalClassif == 1:
                score = classifier.evaluate(feat_test, z_test, verbose=0)
                print(".......... Classification performance")
                #print('Test loss     : %.2f', 100*score[0])
                print('.... Test accuracy : %.2f%%'%(100*score[1]))
            
            # generate a figure
            np.random.seed(110)
            indexes_test = np.random.permutation(x_train.shape[0])

            figName = fileAEModel.replace('.mod','_explesTr.pdf')
            figName = figName.replace('Res/','Res/FIGS/')
            
            if flagOptimMethod == 0:
              figName = figName.replace('.pdf','_Nproj%03d.pdf'%(NbProjection))
            else:
              figName = figName.replace('.pdf','_Nproj%03d_Ngrad%0d.pdf'%(NbProjection,NbGradIter))
        
            idT = int(np.floor(x_train.shape[3]/2))
            plt.figure()
            Nii = 4
            for ii in range(Nii):
                Vmin = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.05 )
                Vmax = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.95 )
                plt.subplot(5, Nii, ii + 1)
                plt.axis('off')
                maskFig = np.copy(mask_train[indexes_test[ii],:,:,idT].squeeze())
                maskFig[ maskFig == 0] = np.float('NaN')
                plt.imshow(x_train[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.title('%d\n' %(indexes_test[ii]))
                plt.subplot(5, Nii, ii + 1+Nii)
                plt.axis('off')
                plt.imshow(x_train_missing[indexes_test[ii],:,:,idT].squeeze() * maskFig, cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, Nii, ii + 1+2*Nii)
                plt.axis('off')
                plt.imshow(x_train_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, Nii, ii + 1+3*Nii)
                plt.axis('off')
                plt.imshow(rec_AE_Tr[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, Nii, ii + 1+4*Nii)
                plt.axis('off')
                if flagloadOIData == 1:
                    plt.imshow(x_train_OI[indexes_test[ii],:,:].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                    
            print('..... Save figure: '+figName)
            plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', quality=98)

            indexes_test = np.random.permutation(x_test.shape[0])
        
            #figName = dirSAVE+'FIGS/'+genFilename+genSuffixModel+'_examplesTt_iter%03d'%(iter)+'.pdf'
            figName = figName.replace('_explesTr','_explesTt')
            
            idT = int(np.floor(x_test.shape[3]/2))
            plt.figure()
            for ii in range(Nii):
                Vmin = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.05 )
                Vmax = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.95 )
                maskFig = np.copy(mask_test[indexes_test[ii],:,:,idT].squeeze())
                maskFig[ maskFig == 0] = np.float('NaN')
                plt.subplot(5, Nii, ii + 1)
                plt.axis('off')
                plt.imshow(x_test[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.title('%d\n GT' %(indexes_test[ii]))
                plt.subplot(5, Nii, ii + 1+Nii)
                plt.axis('off')
                plt.imshow(x_test_missing[indexes_test[ii],:,:,idT].squeeze() * maskFig, cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, Nii, ii + 1+2*Nii)
                plt.axis('off')
                plt.imshow(x_test_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, Nii, ii + 1+3*Nii)
                plt.axis('off')
                plt.imshow(rec_AE_Tt[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, Nii, ii + 1+4*Nii)
                plt.axis('off')
                plt.imshow(maskFig,cmap=plt.cm.gray)
                if flagloadOIData == 1:
                    plt.imshow(x_test_OI[indexes_test[ii],:,:].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                
            print('..... Save figure: '+figName)
            plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', quality=98)
#            np.random.seed(100)
#            indexes_test = np.random.permutation(x_test.shape[0])
#
#            figName = fileAEModel.replace('.mod','_examples.pdf')
#            figName = figName.replace('MNIST/','MNIST/FIGS/')
#            SuffName = '_N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NbProjection))
#            if flagOptimMethod == 1:        
#                SuffName = SuffName + '_NGrad'+str('%02d'%(NbGradIter))
#            figName = figName.replace('.pdf',SuffName+'.pdf')
#                
#            idT = int(np.floor(x_test.shape[3]/2))
#            plt.figure()
#            for ii in range(5):
#                plt.subplot(4, 5, ii + 1)
#                plt.axis('off')
#                plt.imshow(x_test[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
#                plt.title('%d\n GT:%i' %(indexes_test[ii],y_test[indexes_test[ii]]))
#                plt.subplot(4, 5, ii + 1+5)
#                plt.axis('off')
#                plt.imshow(x_test_missing[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
#                plt.subplot(4, 5, ii + 1+10)
#                plt.axis('off')
#                plt.imshow(x_test_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
#                plt.subplot(4, 5, ii + 1+15)
#                plt.axis('off')
#                plt.imshow(rec_AE_Tt[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
#            print('..... Save figure: '+figName)
#            plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
#                        orientation='portrait', quality=98)


        ###############################################################
        ## Performance evaluation for OI 
        elif flagProcess[kk] == 10:        
            NbProjection    = 5   
            NbGradIter      = 0
            flagOptimMethod = 0             
            flagEvalClassif = 0

            # OI covariance parameters
            Lx = 10
            Ly = 10
            Lt = 3
            
            # scaleLR
            scaleLR  = 4
            thrLR    = 1/8
            obsNoise = np.array([0.5])
            
            #### down-sampling and local averaging before applying OI
            #### Keras implementation
            print('.................. Spatial downsampling x%d '%(2**scaleLR))
            x_input         = keras.layers.Input((x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            mask            = keras.layers.Input((x_train.shape[1],x_train.shape[2],x_train.shape[3]))
            
            x   = keras.layers.Lambda(lambda x: x)(x_input)
            m   = keras.layers.Lambda(lambda x: x + 1e-8)(mask)
            for kk in range(0,scaleLR):
                mx = keras.layers.Multiply()([x,m])        
                mx = keras.layers.AveragePooling2D((2,2), padding='valid')(mx)
                m  = keras.layers.AveragePooling2D((2,2), padding='valid')(m)
                x  = keras.layers.Lambda(lambda inputs: inputs[0] / inputs[1])([mx, m])
            
            model_DwSampling = keras.models.Model([x_input,mask],[x,m])
            model_DwSampling.summary()            
            
            
            #### Grid coordinates
            dT    = np.floor(x_train.shape[3]/2).astype(int)
            mat_x = np.array(np.arange(0,x_train.shape[1])).reshape((x_train.shape[1],1))*np.ones((1,x_train.shape[2]))
            mat_y = (np.array(np.arange(0,x_train.shape[2])).reshape((x_train.shape[2]),1)*np.ones((1,x_train.shape[1]))).transpose()
            mat_t = np.array(np.arange(-dT,dT+1))
            
            mat_x = np.tile(mat_x.reshape((x_train.shape[1],x_train.shape[2],1)),(1,1,x_train.shape[3]))
            mat_y = np.tile(mat_y.reshape((x_train.shape[1],x_train.shape[2],1)),(1,1,x_train.shape[3]))
            mat_t = np.tile(mat_t.reshape((1,1,x_train.shape[3])),(x_train.shape[1],x_train.shape[2],1))
            
            #### Apply OI to training dataset
            x_train_OI   = np.zeros((x_train.shape[0],x_train.shape[1],x_train.shape[2]))
            obsLR,maskLR = model_DwSampling.predict([x_train*mask_train,mask_train])
            maskLR       = (maskLR >= thrLR).astype(float)
            obsLR        = obsLR * maskLR            

            mat_x_LR = np.array(np.arange(0,obsLR.shape[1])).reshape((obsLR.shape[1],1))*np.ones((1,obsLR.shape[2]))
            mat_y_LR = (np.array(np.arange(0,obsLR.shape[2])).reshape((obsLR.shape[2]),1)*np.ones((1,obsLR.shape[1]))).transpose()
            mat_t_LR = np.array(np.arange(-5,6))
            
            
            mat_x_LR = np.tile(mat_x_LR.reshape((obsLR.shape[1],obsLR.shape[2],1)),(1,1,obsLR.shape[3]))
            mat_y_LR = np.tile(mat_y_LR.reshape((obsLR.shape[1],obsLR.shape[2],1)),(1,1,obsLR.shape[3]))
            mat_t_LR = np.tile(mat_t_LR.reshape((1,1,obsLR.shape[3])),(obsLR.shape[1],obsLR.shape[2],1))
            
            for ii in range(0,1):#x_train.shape[0]):
                print('.... Apply OI to training data # %d/%d'%(ii,x_train.shape[0]))
                # extraction obsserved pixels
                #ind   = np.where( mask_train[idx,0:-1:step,0:-1:step,:].flatten() == 1)
                ind   = np.where( maskLR[ii,:,:,:].flatten() == 1 )
                obs_x = (2**scaleLR) * mat_x_LR[:,:,:].flatten()[ ind[0][:]]
                obs_y = (2**scaleLR) * mat_y_LR[:,:,:].flatten()[ ind[0][:]]
                obs_t = mat_t_LR[:,:,:].flatten()[ ind[0][:]]
                zobs  = obsLR[:,:,:].flatten()[ind[0][:]]
            
                # Data to be interpolated
                # center of the time window
                ana_x = mat_x[:,:,dT]  
                ana_y = mat_y[:,:,dT] 
                ana_t = mat_t[:,:,dT]
            
                # OI routine
                z_ana,z_std = OI.optimalInterp(obs_x, obs_y, zobs,obs_t,ana_x, ana_y,ana_t, Lx, Ly,Lt,obsNoise)
                z_ana       = z_ana.reshape((x_train.shape[1],x_train.shape[2],1))
            
                # Compute statistics
                x_train_OI[ii,:,:] = z_ana.reshape((1,x_train.shape[1],x_train.shape[2]))
                
            #### Apply OI to training dataset
            #### Apply OI to training dataset
            x_test_OI   = np.zeros((x_test.shape[0],x_train.shape[1],x_train.shape[2]))
            obsLR,maskLR = model_DwSampling.predict([x_test*mask_test,mask_test])
            maskLR       = (maskLR >= thrLR).astype(float)
            obsLR        = obsLR * maskLR            

            mat_x_LR = np.array(np.arange(0,obsLR.shape[1])).reshape((obsLR.shape[1],1))*np.ones((1,obsLR.shape[2]))
            mat_y_LR = (np.array(np.arange(0,obsLR.shape[2])).reshape((obsLR.shape[2]),1)*np.ones((1,obsLR.shape[1]))).transpose()
            mat_t_LR = np.array(np.arange(-5,6))
            
            
            mat_x_LR = np.tile(mat_x_LR.reshape((obsLR.shape[1],obsLR.shape[2],1)),(1,1,obsLR.shape[3]))
            mat_y_LR = np.tile(mat_y_LR.reshape((obsLR.shape[1],obsLR.shape[2],1)),(1,1,obsLR.shape[3]))
            mat_t_LR = np.tile(mat_t_LR.reshape((1,1,obsLR.shape[3])),(obsLR.shape[1],obsLR.shape[2],1))

            for ii in range(0,x_test.shape[0]):
                print('.... Apply OI to test data # %d/%d'%(ii,x_test.shape[0]))
                # extraction obsserved pixels
                #ind   = np.where( mask_train[idx,0:-1:step,0:-1:step,:].flatten() == 1)
                ind   = np.where( maskLR[ii,:,:,:].flatten() == 1 )
                obs_x = (2**scaleLR) * mat_x_LR[:,:,:].flatten()[ ind[0][:]]
                obs_y = (2**scaleLR) * mat_y_LR[:,:,:].flatten()[ ind[0][:]]
                obs_t = mat_t_LR[:,:,:].flatten()[ ind[0][:]]
                zobs  = obsLR[:,:,:].flatten()[ind[0][:]]
            
                # Data to be interpolated
                # center of the time window
                ana_x = mat_x[:,:,dT]  
                ana_y = mat_y[:,:,dT] 
                ana_t = mat_t[:,:,dT]
            
                # OI routine
                z_ana,z_std = OI.optimalInterp(obs_x, obs_y, zobs,obs_t,ana_x, ana_y,ana_t, Lx, Ly,Lt,obsNoise)
                z_ana       = z_ana.reshape((x_train.shape[1],x_train.shape[2],1))
            
                # Compute statistics
                x_test_OI[ii,:,:] = z_ana.reshape((1,x_train.shape[1],x_train.shape[2]))
             
                # Compute statistics
                x_train_OI[ii,:,:] = z_ana.reshape((1,x_train.shape[1],x_train.shape[2]))

            mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp = eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_OI,
                                       mask_test,x_test,x_test_missing,x_test_OI)
            
            
            #### Performance Summary
            print(".......... Reconstruction performance with Missing Data")
            print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
            print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
            print('....')
            print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
            print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
            print('....')
            print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
            print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))
                            
            # generate a figure
            np.random.seed(100)
            indexes_test = np.random.permutation(x_train.shape[0])

            figName = fileAEModel.replace('.mod','_explesTr.pdf')
            figName = figName.replace('Res/','Res/FIGS/')
            
            figName = figName.replace('.pdf','_OI.pdf'%(NbProjection))
        
            idT = int(np.floor(x_train.shape[3]/2))
            plt.figure()
            for ii in range(3):
                Vmin = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.05 )
                Vmax = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.95 )
                plt.subplot(4, 3, ii + 1)
                plt.axis('off')
                maskFig = np.copy(mask_train[indexes_test[ii],:,:,idT].squeeze())
                maskFig[ maskFig == 0] = np.float('NaN')
                plt.imshow(x_train[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.title('%d\n' %(indexes_test[ii]))
                plt.subplot(4, 3, ii + 1+3)
                plt.axis('off')
                plt.imshow(x_train_missing[indexes_test[ii],:,:,idT].squeeze() * maskFig, cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(4, 3, ii + 1+6)
                plt.axis('off')
                plt.imshow(x_train_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(4, 3, ii + 1+9)
                plt.axis('off')
                plt.imshow(rec_AE_Tr[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
            print('..... Save figure: '+figName)
            plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', quality=98)

            indexes_test = np.random.permutation(x_test.shape[0])
        
            #figName = dirSAVE+'FIGS/'+genFilename+genSuffixModel+'_examplesTt_iter%03d'%(iter)+'.pdf'
            figName = figName.replace('_explesTr','_explesTt')
            
            idT = int(np.floor(x_test.shape[3]/2))
            plt.figure()
            for ii in range(3):
                Vmin = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.05 )
                Vmax = np.quantile(x_train[indexes_test[ii],:,:,idT].flatten() , 0.95 )
                maskFig = np.copy(mask_test[indexes_test[ii],:,:,idT].squeeze())
                maskFig[ maskFig == 0] = np.float('NaN')
                plt.subplot(5, 3, ii + 1)
                plt.axis('off')
                plt.imshow(x_test[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.title('%d\n GT' %(indexes_test[ii]))
                plt.subplot(5, 3, ii + 1+3)
                plt.axis('off')
                plt.imshow(x_test_missing[indexes_test[ii],:,:,idT].squeeze() * maskFig, cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, 3, ii + 1+6)
                plt.axis('off')
                plt.imshow(x_test_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, 3, ii + 1+9)
                plt.axis('off')
                plt.imshow(rec_AE_Tt[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.hot,vmin=Vmin,vmax=Vmax)
                plt.subplot(5, 3, ii + 1+12)
                plt.axis('off')
                plt.imshow(maskFig,cmap=plt.cm.gray)
                
            print('..... Save figure: '+figName)
            plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', quality=98)
#            np.random.seed(100)
#            indexes_test = np.random.permutation(x_test.shape[0])
#
#            figName = fileAEModel.replace('.mod','_examples.pdf')
#            figName = figName.replace('MNIST/','MNIST/FIGS/')
#            SuffName = '_N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NbProjection))
#            if flagOptimMethod == 1:        
#                SuffName = SuffName + '_NGrad'+str('%02d'%(NbGradIter))
#            figName = figName.replace('.pdf',SuffName+'.pdf')
#                
#            idT = int(np.floor(x_test.shape[3]/2))
#            plt.figure()
#            for ii in range(5):
#                plt.subplot(4, 5, ii + 1)
#                plt.axis('off')
#                plt.imshow(x_test[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
#                plt.title('%d\n GT:%i' %(indexes_test[ii],y_test[indexes_test[ii]]))
#                plt.subplot(4, 5, ii + 1+5)
#                plt.axis('off')
#                plt.imshow(x_test_missing[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
#                plt.subplot(4, 5, ii + 1+10)
#                plt.axis('off')
#                plt.imshow(x_test_pred[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
#                plt.subplot(4, 5, ii + 1+15)
#                plt.axis('off')
#                plt.imshow(rec_AE_Tt[indexes_test[ii],:,:,idT].squeeze(), cmap=plt.cm.gray_r)
#            print('..... Save figure: '+figName)
#            plt.savefig(figName, dpi=None, facecolor='w', edgecolor='w',
#                        orientation='portrait', quality=98)

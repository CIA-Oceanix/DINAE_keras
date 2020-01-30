from dinae import *
from .tools import *
from .graphics import *

def thresholding(x,thr):    
    greater = K.greater_equal(x,thr) #will return boolean values
    greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1    
    return greater

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
    exp_var_train  = 1. - mse_train #/ var_Tr
            
    mse_test        = np.zeros((2))
    mse_test[0]     = np.sum( mask_test * (x_test_pred - x_test_missing)**2 ) / np.sum( mask_test )
    mse_test[1]     = np.mean( (x_test_pred - x_test)**2 ) 
    exp_var_test = 1. - mse_test #/ var_Tt

    mse_train_interp        = np.sum( (1.-mask_train) * (x_train_pred - x_train)**2 ) / np.sum( 1. - mask_train )
    exp_var_train_interp    = 1. - mse_train_interp 
    
    mse_test_interp        = np.sum( (1.-mask_test) * (x_test_pred - x_test)**2 ) / np.sum( 1. - mask_test )
    exp_var_test_interp    = 1. - mse_test_interp
            
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
        maskg = keras.layers.Lambda(lambda x: thresholding(x,thrNoise))(maskg)    
     
        maskg  = keras.layers.Multiply()([mask,maskg])
        maskg  = keras.layers.Subtract()([mask,maskg])       
    else:
        maskg = keras.layers.Lambda(lambda x: 1.*x)(mask)
      
    if flag_MultiScaleAEModel == 0:
        x_proj = global_model_FP([x_input,maskg])
    else:
        x_proj,x_projLR = global_model_FP_MR([x_input,maskg])
  
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
  
    x = keras.layers.Conv2D(shape[3],(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
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
        gradModel  = keras.models.Model([x_input],[x])
      
    elif flagGradModel == 2:
        print("...... Initialize Gradient Model: LSTM")
        x_input  = keras.layers.Input((shape[1],shape[2],shape[3]*2))

        gx_lstm  = keras.layers.ConvLSTM2D(10, kernel_size=(1,3), padding='same', use_bias=False, activation='relu')(keras.layers.Reshape((1,shape[1],shape[2],2))(x_input))#,return_state=True)#,input_shape=(1, para_Model.NbHS))
        gx_lstm  = keras.layers.Conv2D(shape[3],(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(gx_lstm)
      
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
          
 
        maskg = keras.layers.Lambda(lambda x: thresholding(x,thrNoise))(maskg)    
     
        maskg  = keras.layers.Multiply()([mask,maskg])
        maskg  = keras.layers.Subtract()([mask,maskg])       
    else:
        maskg = keras.layers.Lambda(lambda x: 1.*x)(mask)

    x_proj = global_model_Grad([x_input,maskg])
  
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

    if flagDisplay == 1:
        gradModel.summary()
        gradMaskModel.summary()
        global_model_Grad.summary()
        global_model_Grad_Masked.summary()
  
    return global_model_Grad,global_model_Grad_Masked

def flagProcess4(dict_global_Params,genFilename,x_train,x_train_missing,mask_train,meanTr,stdTr,x_test,x_test_missing,mask_test,lday_test,encoder,decoder,model_AE,DimCAE):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    # PCA decomposition for comparison
    pca      = decomposition.PCA(DimCAE)
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
        
        # generate some plots
        figpathTr = dirSAVE+'FIGS/Iter_%03d'%(iter)+'_Tr'
        if not os.path.exists(figpathTr):
            mk_dir_recursive(figpathTr)
        else:
            shutil.rmtree(figpathTr)
            mk_dir_recursive(figpathTr) 
        figpathTt = dirSAVE+'FIGS/Iter_%03d'%(iter)+'_Tt'
        if not os.path.exists(figpathTt):
            mk_dir_recursive(figpathTt)
        else:
            shutil.rmtree(figpathTt)
            mk_dir_recursive(figpathTt) 

        idT = 0
        lon = np.arange(-65,-55,1/20)
        lat = np.arange(30,40,1/20)
        indLat     = np.arange(0,200)
        indLon     = np.arange(0,200)
        lon = lon[indLon]
        lat = lat[indLat]
        extent_=[np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
        lfig=[20,40,60]
        # Training dataset
        for ifig in lfig:

            # Rough variables
            figName = figpathTr+'/'+genFilename+genSuffixModel+'_examplesTr_%03d'%(ifig)+'.png' 
            fig, ax = plt.subplots(2,2,figsize=(15,15),
                          subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
            vmin = np.quantile(x_train[ifig,:,:,idT].flatten() , 0.05 )
            vmax = np.quantile(x_train[ifig,:,:,idT].flatten() , 0.95 )
            cmap="coolwarm"
            GT   = x_train[ifig,:,:,idT].squeeze()
            OBS  = np.where(x_train_missing[ifig,:,:,idT].squeeze()==0,\
                     np.nan, x_train_missing[ifig,:,:,idT].squeeze())
            PRED = x_train_pred[ifig,:,:,idT].squeeze()
            REC  = rec_AE_Tr[ifig,:,:,idT].squeeze()
            plot(ax,0,0,lon,lat,GT,"GT",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,0,1,lon,lat,OBS,"Observations",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,1,0,lon,lat,PRED,"Pred",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,1,1,lon,lat,REC,"Rec",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plt.subplots_adjust(hspace=0.5,wspace=0.25)
            plt.savefig(figName)       # save the figure
            plt.close()                # close the figure

            # Gradient
            figName = figpathTr+'/'+genFilename+genSuffixModel+'_examplesTr_grads_%03d'%(ifig)+'.png'
            fig, ax = plt.subplots(2,2,figsize=(15,15),
                          subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
            vmin = np.quantile(Gradient(x_train[ifig,:,:,idT],2).flatten() , 0.05 )
            vmax = np.quantile(Gradient(x_train[ifig,:,:,idT],2).flatten() , 0.95 )
            cmap="viridis"
            GT   = Gradient(x_train[ifig,:,:,idT].squeeze(),2)
            OBS  = Gradient(np.where(x_train_missing[ifig,:,:,idT].squeeze()==0,\
                     np.nan,x_train_missing[ifig,:,:,idT].squeeze()),2)
            PRED = Gradient(x_train_pred[ifig,:,:,idT].squeeze(),2)
            REC  = Gradient(rec_AE_Tr[ifig,:,:,idT].squeeze(),2)
            plot(ax,0,0,lon,lat,GT,r"$\nabla_{GT}$",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,0,1,lon,lat,OBS,r"$\nabla_{Obs}$",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,1,0,lon,lat,PRED,r"$\nabla_{Pred}$",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,1,1,lon,lat,REC,r"$\nabla_{Rec}$",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plt.subplots_adjust(hspace=0.5,wspace=0.25)
            plt.savefig(figName)       # save the figure
            plt.close()                # close the figure

        # Test dataset
        lfig=[15,30,45]
        for ifig in lfig:

            # Rough variables
            figName = figpathTt+'/'+genFilename+genSuffixModel+'_examplesTt_%03d'%(ifig)+'_'+lday_test[ifig]+'.png'
            fig, ax = plt.subplots(2,2,figsize=(15,15),
                          subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
            vmin = np.quantile(x_test[ifig,:,:,idT].flatten() , 0.05 )
            vmax = np.quantile(x_test[ifig,:,:,idT].flatten() , 0.95 )
            cmap="coolwarm"
            GT   = x_test[ifig,:,:,idT].squeeze()
            OBS  = np.where(x_test_missing[ifig,:,:,idT].squeeze()==0,\
                     np.nan, x_test_missing[ifig,:,:,idT].squeeze())
            PRED = x_test_pred[ifig,:,:,idT].squeeze()
            REC  = rec_AE_Tt[ifig,:,:,idT].squeeze()
            plot(ax,0,0,lon,lat,GT,"GT",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,0,1,lon,lat,OBS,"Observations",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,1,0,lon,lat,PRED,"Pred",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,1,1,lon,lat,REC,"Rec",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plt.subplots_adjust(hspace=0.5,wspace=0.25)
            plt.savefig(figName)       # save the figure
            plt.close()                # close the figure

            # Gradient variables
            figName = figpathTt+'/'+genFilename+genSuffixModel+'_examplesTt_grads_%03d'%(ifig)+'_'+lday_test[ifig]+'.png'
            fig, ax = plt.subplots(2,2,figsize=(15,15),
                          subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0.0)))
            vmin = np.quantile(Gradient(x_test[ifig,:,:,idT],2).flatten() , 0.05 )
            vmax = np.quantile(Gradient(x_test[ifig,:,:,idT],2).flatten() , 0.95 )
            cmap="viridis"
            GT   = Gradient(x_test[ifig,:,:,idT].squeeze(),2)
            OBS  = Gradient(np.where(x_test_missing[ifig,:,:,idT].squeeze()==0,\
                     np.nan, x_test_missing[ifig,:,:,idT].squeeze()),2)
            PRED = Gradient(x_test_pred[ifig,:,:,idT].squeeze(),2)
            REC  = Gradient(rec_AE_Tt[ifig,:,:,idT].squeeze(),2)
            plot(ax,0,0,lon,lat,GT,r"$\nabla_{GT}$",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,0,1,lon,lat,OBS,r"$\nabla_{Observations}$",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,1,0,lon,lat,PRED,r"$\nabla_{Pred}$",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plot(ax,1,1,lon,lat,REC,r"$\nabla_{Rec}$",\
                 extent=extent_,cmap=cmap,vmin=vmin,vmax=vmax)
            plt.subplots_adjust(hspace=0.5,wspace=0.25)
            plt.savefig(figName)       # save the figure
            plt.close()                # close the figure


        # Save DINAE result         
        saved_path = dirSAVE+'/saved_path_%03d'%(iter)+'.pickle'
        with open(saved_path, 'wb') as handle:
            pickle.dump([x_test,x_test_missing,x_test_pred,rec_AE_Tt], handle)


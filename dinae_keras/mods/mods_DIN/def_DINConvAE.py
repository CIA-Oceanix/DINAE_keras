from dinae_keras import *
from ..tools import *

def slice_layer(index):
    def func(x_input):
        return tf.gather(x_input, index, axis=3)
    return keras.layers.Lambda(func)

def assign_sliced_layer(size_tw,N_cov,x_output):
    def func(x_input,x_output):
        for i in range(1,(N_cov+1)):
            index = np.arange(i,(N_cov+1)*size_tw,(N_cov+1),dtype='int32')
            x_output = keras.layers.Concatenate()([x_output,slice_layer(index)(x_input)])
        index  = np.stack([np.arange(i*size_tw,(i+1)*size_tw) \
                           for i in range(0,N_cov+1)]).T.flatten()
        x_proj = slice_layer(index)(x_output)
        return x_proj
    return keras.layers.Lambda(func,arguments={'x_output':x_output})

def error(x1,x2,mask,size_tw,shape,alpha,N_cov):
    if x1.shape[3]>x2.shape[3]:
        index = np.arange(0,(N_cov+1)*size_tw,N_cov+1,dtype='int32')
        err   = keras.layers.Subtract()([slice_layer(index)(x1),x2])
        err   = keras.layers.Multiply()([err,slice_layer(index)(mask)])
    elif x2.shape[3]>x1.shape[3]:
        index = np.arange(0,(N_cov+1)*size_tw,N_cov+1,dtype='int32')
        err   = keras.layers.Subtract()([x1,slice_layer(index)(x2)])
        err   = keras.layers.Multiply()([err,slice_layer(index)(mask)])
    else:
        err   = keras.layers.Subtract()([x1,x2])
        err   = keras.layers.Multiply()([err,mask])
    err   = keras.layers.Multiply()([err,err])
    err   = keras.layers.Reshape((err.shape[-3],err.shape[-2],err.shape[-1],1))(err)
    # normalize err
    err   = keras.layers.GlobalAveragePooling3D()(err)
    err   = keras.layers.Reshape((1,))(err)
    err   = keras.layers.Lambda(lambda x: alpha*x)(err)
    return err

def regularize_Gradient(x_proj,size_tw):
    filter_gx  = K.constant(insert_Sobel(size_tw,"x"))
    filter_gy  = K.constant(insert_Sobel(size_tw,"y"))
    def Gradient(tensors):
        tensor = tensors[0]
        filter = tensors[1]
        return K.conv2d(tensor, filter, padding="same")
    Gx_pred = keras.layers.Lambda(Gradient)([x_proj, filter_gx])
    Gy_pred = keras.layers.Lambda(Gradient)([x_proj, filter_gy])
    #Grad_pred = K.sqrt(keras.layers.Add()([keras.layers.Multiply()([Gx_pred,Gx_pred]),\
    #                                    keras.layers.Multiply()([Gy_pred,Gy_pred])]))
    Grad_pred = keras.layers.Add()([keras.layers.Multiply()([Gx_pred,Gx_pred]),\
                                        keras.layers.Multiply()([Gy_pred,Gy_pred])])
    reg_Gradient   = keras.layers.Reshape((Grad_pred.shape[-3],Grad_pred.shape[-2],Grad_pred.shape[-1],1))(Grad_pred)
    reg_Gradient   = keras.layers.GlobalAveragePooling3D()(reg_Gradient)
    reg_Gradient   = keras.layers.Reshape((1,))(reg_Gradient)
    reg_Gradient   = keras.layers.Lambda(lambda x: 1*x)(reg_Gradient)
    return reg_Gradient

def define_DINConvAE(NiterProjection,model_AE,shape,\
                     flag_MultiScaleAEModel,flagUseMaskinEncoder,\
                     size_tw,include_covariates,N_cov=0):

    # encoder-decoder with masked data
    x_input         = keras.layers.Input((shape[1],shape[2],shape[3]))
    mask            = keras.layers.Input((shape[1],shape[2],shape[3]))
    
    x     = keras.layers.Lambda(lambda x:1.*x)(x_input)
    mask_ = keras.layers.Lambda(lambda x:1.-x)(mask)

    # Iterations of fixed-point projection
    index = np.arange(0,(N_cov+1)*size_tw,N_cov+1)   
    for kk in range(0,NiterProjection):
        x_proj   = model_AE([x,mask])
        x_proj   = keras.layers.Multiply()([x_proj,slice_layer(index)(mask_)])
        x        = keras.layers.Multiply()([slice_layer(index)(x),slice_layer(index)(mask)])
        x        = keras.layers.Add()([x,x_proj])
        if include_covariates==True:
            x = assign_sliced_layer(size_tw,N_cov,x)(x_input)

    if flag_MultiScaleAEModel == 1:
        x_proj,x_projLR = model_AE_MR([x,mask])
        global_model_FP    = keras.models.Model([x_input,mask],[x_proj])
        global_model_FP_MR = keras.models.Model([x_input,mask],[x_proj,x_projLR])
    else:
        x_proj = model_AE([x,mask]) 
        global_model_FP    = keras.models.Model([x_input,mask],[x_proj])

    # randomly sample an additionnal missing data mask
    # additive noise + spatial smoothing
    if flagUseMaskinEncoder == 1:
        WAvFilter     = 3
        NIterAvFilter = 3
        thrNoise      = 1.5 * stdMask + 1e-7
        maskg   = keras.layers.GaussianNoise(stdMask)(mask)
        avFilter       = 1./(WAvFilter**3)*np.ones((WAvFilter,WAvFilter,WAvFilter,1,1))
        spatialAvLayer = keras.layers.Conv3D(1,(WAvFilter,WAvFilter,WAvFilter),weights=[avFilter],\
                           padding='same',activation='linear',use_bias=False,name='SpatialAverage')
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
    err1 = error(x_proj,x_input,mask,size_tw,shape,1,N_cov)
    # AE error with x_proj
    if flag_MultiScaleAEModel == 1:
        err1LR = error(x_projLR,x_input,mask,size_tw,shape,0,N_cov)
        err1   = keras.layers.Add()([err1,err1LR])
    # compute error (x_proj-x_input)**2 with full-1 mask
    x_proj_ = x_proj
    if include_covariates==False:
        x_proj2 = model_AE([x_proj_,keras.layers.Lambda(lambda x:1.-0.*x)(mask)])
    else:
        index  = np.arange(0,(N_cov+1)*size_tw,N_cov+1,dtype='int32')
        x_proj_.set_shape((x_input.shape[0],x_input.shape[1],\
                          x_input.shape[2],size_tw))
        x_proj_ = assign_sliced_layer(size_tw,N_cov,x_proj_)(x_input)
        x_proj_ = x_input
        x_proj2 = model_AE([x_proj_,keras.layers.Lambda(lambda x:1.-0.*x)(mask)])
    err2    = error(x_proj_,x_proj2,mask,size_tw,shape,0,N_cov)
    # compute error (x_proj-x_input)**2 with full-1 mask
    x_proj3 = model_AE([x_proj_,keras.layers.Lambda(lambda x:0.*x)(mask)])
    err3    = error(x_proj3,x_proj_,mask,size_tw,shape,0,N_cov)
    # add all errors
    err    = keras.layers.Add()([err1,err2])
    err    = keras.layers.Add()([err,err3])

    # return global model
    global_model_FP_Masked  = keras.models.Model([x_input,mask],[err,x_proj])

    global_model_FP.summary()
    global_model_FP_Masked.summary()
  
    return global_model_FP,global_model_FP_Masked


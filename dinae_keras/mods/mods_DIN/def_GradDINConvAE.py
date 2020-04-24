from dinae_keras import *
 
def error(x1,x2,mask,shape,alpha):
    err   = keras.layers.Subtract()([x1,x2])
    err   = keras.layers.Multiply()([err,mask])
    err   = keras.layers.Multiply()([err,err])
    err   = keras.layers.Reshape((shape[1],shape[2],shape[3],1))(err)
    err   = keras.layers.GlobalAveragePooling3D()(err)
    err   = keras.layers.Reshape((1,))(err)
    err   = keras.layers.Lambda(lambda x: alpha*x)(err)
    return err 

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
    err1 = error(x_proj,x_input,mask,shape,1)
    # compute error (x_proj-x_input)**2 with full-1 mask
    x_proj2 = model_AE([x_proj,keras.layers.Lambda(lambda x:1.-0.*x)(mask)])
    err2    = error(x_proj,x_proj2,mask,shape,0)
    # compute error (x_proj-x_input)**2 with full-1 mask
    x_proj3 = model_AE([x_proj,keras.layers.Lambda(lambda x:0.*x)(mask)])
    err3    = error(x_proj3,x_proj,mask,shape,0)
    # add all errors
    err    = keras.layers.Add()([err1,err2])
    err    = keras.layers.Add()([err,err3])

    # Models and print summary
    global_model_Grad_Masked  = keras.models.Model([x_input,mask],err)
    gradModel.summary()
    gradMaskModel.summary()
    global_model_Grad.summary()
    global_model_Grad_Masked.summary()
  
    return global_model_Grad,global_model_Grad_Masked


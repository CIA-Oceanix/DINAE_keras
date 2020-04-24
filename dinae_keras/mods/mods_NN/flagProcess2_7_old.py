from dinae_keras import *

# Constraints on kernel to to zeros
# a specific position
class Constraint_Zero(Constraint):
    def __init__(self, position,kernel_shape,dw):
        self.position = position
        mask_array    = np.ones((kernel_shape))
        mask_array[position[0]-dw:position[0]+dw+1,position[1]-dw:position[1]+dw+1,:,:] = 0.0

        self.mask = K.variable(value=mask_array, dtype='float32', name='mask')

        print(self.mask.shape)
    def __call__(self, w):
        print(self.mask.shape)
        new_w = w * self.mask

        return new_w

def flagProcess2_7(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    WFilter       = 11#
    NbResUnit     = 10#3#
    dW    = 0
    flagdownScale = 1 #: 0: only HR scale, 1 : only LR, 2 : HR + LR , 2 : MR, HR + LR annd LR,
    scaleLR       = 2**2
    NbFilter      = 1*DimAE
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
                dx = keras.layers.Conv2D(5*DimAE,(1,1),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
            else:
                dx = keras.layers.Lambda(lambda x: scale * x)(x)
      
            dx_lin  = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
            dx1 = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
            dx2 = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
      
            dx1 = keras.layers.Multiply()([dx1,dx2])
            
            dx  = keras.layers.Add()([dx1,dx_lin])
            dx  = keras.layers.Activation('tanh')(dx)
            x  = keras.layers.Add()([x,dx])
  
        x  = keras.layers.Conv2D(x_train.shape[3],(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)          
        x1 = keras.layers.Conv2DTranspose(x_train.shape[3],(scaleLR,scaleLR),strides=(scaleLR,scaleLR),use_bias=False,activation='linear',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
        
        if flagSRResNet == 1: ## postprocessing: super-resolution-like block
            x1  = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)               
            
            scale = 0.1
            for kk in range(0,NbResUnit):
                if 1*1 :
                    dx = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)
                else:
                    dx = keras.layers.Lambda(lambda x: scale * x)(x1)
          
                dx_lin  = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                dx1 = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                dx2 = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
          
                dx1 = keras.layers.Multiply()([dx1,dx2])
        
                dx  = keras.layers.Add()([dx1,dx_lin])
                dx  = keras.layers.Activation('tanh')(dx_lin)
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
        dx  = keras.layers.Activation('tanh')(dx)
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
        
        input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
        mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
        x,xLR      = encoder([input_data,mask])
        
        model_AE    = keras.models.Model([input_data,mask],x)
        model_AE_MR = keras.models.Model([input_data,mask],[x,xLR])

    model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
    model_AE.summary()

    DimCAE = DimAE

    return genFilename, encoder, decoder, model_AE, DimCAE

from dinae_keras import *
from ..tools import keras_custom_loss_function

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

def GENN(dict_global_Params,flagdownScale,flagSRResNet,genFilename,x_data,mask_data):
    # flagdownScale = 0: only HR scale
    # flagdownScale = 1: only LR scale
    # flagdownScale = 2: HR + LR scales
    # flagdownScale = 3: MR, HR + LR and LR scales

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    WFilter       = 11
    NbResUnit     = 10
    dW            = 0
    scaleLR       = 2**2
    NbFilter      = 1*DimAE

    # define generic Filename
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

    # define input and mask layers
    input_layer = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    mask       = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))

    # use Mask in Encoder
    if flagUseMaskinEncoder == 1:
        dmask   = keras.layers.Lambda(lambda x: 0.2*x - 0.1)(mask)
        for jj in range(0,6):
            dx    = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dmask)            
            dx    = keras.layers.Conv2D(1,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)            
            dmask = keras.layers.Add()([dmask,dx]) 
        x0       = keras.layers.Concatenate(axis=-1)([input_layer,dmask])
        mx	 = 2
    else:
        x0  = keras.layers.Lambda(lambda x: 1. * x)(input_layer)
        mx  = 1

    # coarse scale (xLR)
    if flagdownScale > 0 :
        x = keras.layers.AveragePooling2D((scaleLR,scaleLR), padding='valid')(x0)
        x = keras.layers.Conv2D(NbFilter,(WFilter,WFilter),activation='relu', padding='same',use_bias=False,
            			kernel_regularizer=keras.regularizers.l2(wl2),
            			kernel_constraint=Constraint_Zero((int(WFilter/2),int(WFilter/2)),(WFilter,WFilter,mx*x_data.shape[3],NbFilter),dW))(x)
        x = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
        # registration/evolution in feature space
        scale = 0.1
        for kk in range(0,NbResUnit):
            dx = keras.layers.Conv2D(5*DimAE,(1,1),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
            dx_lin  = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
            dx1 = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
            dx2 = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
            dx1 = keras.layers.Multiply()([dx1,dx2])
            dx  = keras.layers.Add()([dx1,dx_lin])
            dx  = keras.layers.Activation('tanh')(dx)
            x  = keras.layers.Add()([x,dx])
        x  = keras.layers.Conv2D(int(x_data.shape[3]/(N_cov+1)),(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
        x1 = keras.layers.Conv2DTranspose(int(x_data.shape[3]/(N_cov+1)),(scaleLR,scaleLR),strides=(scaleLR,scaleLR),use_bias=False,activation='linear',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
        # postprocessing: super-resolution-like block
        if flagSRResNet == 1: 
            x1  = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)               
            scale = 0.1
            for kk in range(0,NbResUnit):
                dx = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)
                dx_lin  = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                dx1 = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                dx2 = keras.layers.Conv2D(DimAE,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
                dx1 = keras.layers.Multiply()([dx1,dx2])
                dx  = keras.layers.Add()([dx1,dx_lin])
                dx  = keras.layers.Activation('tanh')(dx_lin)
                x1  = keras.layers.Add()([x1,dx])
            xLR  = keras.layers.Conv2D(x_data.shape[3]/(N_cov+1),(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)
        else:
            x1  = keras.layers.Conv2D(2*DimAE,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1) 
            xLR  = keras.layers.Conv2D(int(x_data.shape[3]/(N_cov+1)),(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x1)          
     
    # fine scale (xHR)
    x = keras.layers.Conv2D(NbFilter,(WFilter,WFilter),activation='relu', 
            padding='same',use_bias=False,
            kernel_regularizer=keras.regularizers.l2(wl2),
            kernel_constraint=Constraint_Zero((int(WFilter/2),int(WFilter/2)),(WFilter,WFilter,mx*x_data.shape[3],NbFilter),dW))(x0)
    if flagdownScale > 0 :
        x  = keras.layers.Concatenate(axis=-1)([x,x1])
    x  = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    for kk in range(0,NbResUnit):
        dx      = keras.layers.Conv2D(5*DimAE,(1,1),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x) 
        dx_lin  = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
        dx1 = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
        dx2 = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx) 
        dx1 = keras.layers.Multiply()([dx1,dx2])
        dx  = keras.layers.Add()([dx1,dx_lin])
        dx  = keras.layers.Activation('tanh')(dx)
        x  = keras.layers.Add()([x,dx])
    xHR  = keras.layers.Conv2D(int(x_data.shape[3]/(N_cov+1)),(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)

    ## Building encoder/decoder
    # build encoder
    if flagdownScale == 0:
        encoder    = keras.models.Model([input_layer,mask],xHR)
    elif flagdownScale == 1:
        encoder    = keras.models.Model([input_layer,mask],xLR)
    elif flagdownScale == 2:
        x          = keras.layers.Add()([xHR,xLR]) 
        encoder    = keras.models.Model([input_layer,mask],x)
    elif flagdownScale == 3:
        x          = keras.layers.Add()([xHR,xLR]) 
        encoder    = keras.models.Model([input_layer,mask],[xLR,x])
    # build decoder
    if flagdownScale != 3:
        decoder_input = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))    
        x  = keras.layers.Lambda(lambda x: 1. * x)(decoder_input)
        decoder       = keras.models.Model(decoder_input,x)
    else:
        decoder_input1 = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))    
        decoder_input2 = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))    
        x1  = keras.layers.Lambda(lambda x: 1. * x)(decoder_input1)
        x2  = keras.layers.Lambda(lambda x: 1. * x)(decoder_input2)
        decoder       = keras.models.Model([decoder_input1,decoder_input2],[x1,x2])
    encoder.summary()
    decoder.summary()

    ## Building model_AE
    input_data = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    mask       = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    if flagdownScale != 3:
        x          = decoder(encoder([input_data,mask]))
        model_AE   = keras.models.Model([input_data,mask],x)  
    else: 
        x,xLR      = encoder([input_data,mask])
        model_AE    = keras.models.Model([input_data,mask],x)
        model_AE_MR = keras.models.Model([input_data,mask],[x,xLR])   
    model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
    #model_AE.compile(loss=keras_custom_loss_function(int(x_data.shape[3]/(N_cov+1))),optimizer=keras.optimizers.Adam(lr=1e-3))
    model_AE.summary()

    return genFilename, encoder, decoder, model_AE, DimAE




from dinae_keras import *
from ..tools import keras_custom_loss_function

class Constraint_AD(Constraint):
    def __init__(self,kernel_shape):
        self.kernel_shape = kernel_shape
    def __call__(self, w):
        new_w = w 
        new_w_0_0 = K.expand_dims(K.expand_dims(new_w[2,2,:,:],axis=0),axis=0)
        new_w_0_1 = K.expand_dims(K.expand_dims(new_w[2,1,:,:],axis=0),axis=0)
        new_w_0_2 = K.expand_dims(K.expand_dims(-1.*new_w[0,0,:,:],axis=0),axis=0)
        new_w_1_0 = K.expand_dims(K.expand_dims(new_w[1,2,:,:],axis=0),axis=0)
        new_w_1_2 = K.expand_dims(K.expand_dims(new_w[1,2,:,:],axis=0),axis=0)
        new_w_2_0 = K.expand_dims(K.expand_dims(-1.*new_w[0,0,:,:],axis=0),axis=0)
        new_w_2_1 = K.expand_dims(K.expand_dims(new_w[2,1,:,:],axis=0),axis=0)
        new_w_2_2 = K.expand_dims(K.expand_dims(new_w[2,2,:,:],axis=0),axis=0)
        # central point
        new_w_1_1 = K.expand_dims(K.expand_dims(-2.*(new_w[0,1,:,:]+new_w[1,0,:,:]),axis=0),axis=0) #+K2
        first_row  = K.concatenate([new_w_0_0,new_w_0_1,new_w_0_2],axis=1)
        second_row = K.concatenate([new_w_1_0,new_w_1_1,new_w_1_2],axis=1)
        third_row  = K.concatenate([new_w_2_0,new_w_2_1,new_w_2_2],axis=1)
        new_w      = K.concatenate([first_row,second_row,third_row], axis=0)
        return new_w

def PINN(dict_global_Params,genFilename,x_data,mask_data):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    NbResUnit     = 10

    # define generic Filename
    genFilename = genFilename+str('RU%03d_'%NbResUnit)   

    # define input and mask layers
    input_layer = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    mask       = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))

    x = keras.layers.Conv2D(x_data.shape[3],(3,3),activation='relu', 
            padding='same',use_bias=False,
            kernel_regularizer=keras.regularizers.l2(wl2),
            kernel_constraint=Constraint_AD((3,3,x_data.shape[3],x_data.shape[3])))(input_layer)
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
    x  = keras.layers.Conv2D(int(x_data.shape[3]/(N_cov+1)),(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)

    ## Building encoder/decoder
    # build encoder
    encoder    = keras.models.Model([input_layer,mask],x)
    # build decoder
    decoder_input = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))    
    x  = keras.layers.Lambda(lambda x: 1. * x)(decoder_input)
    decoder       = keras.models.Model(decoder_input,x)
    encoder.summary()
    decoder.summary()

    ## Building model_AE
    input_data = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    mask       = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    x          = decoder(encoder([input_data,mask]))
    model_AE   = keras.models.Model([input_data,mask],x)  
    model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
    model_AE.summary()

    return genFilename, encoder, decoder, model_AE, DimAE




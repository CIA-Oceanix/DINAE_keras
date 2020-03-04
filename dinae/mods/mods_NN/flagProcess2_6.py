from dinae import *
from ..tools import keras_custom_loss_function

def flagProcess2_6(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    DimCAE = DimAE

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
    #x = keras.layers.Dropout(dropout)(x)
    #x = keras.layers.AveragePooling2D((2,2), padding='valid')(x)
    #x = keras.layers.Conv2D(16*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.AveragePooling2D((5,5), padding='valid')(x)
    x = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
    
    encoder    = keras.models.Model([input_layer,mask],x)
                 
    decoder_input = keras.layers.Input(shape=(int(np.floor(x_train.shape[1]/40)),int(np.floor(x_train.shape[2]/40)),DimAE))    
    x = keras.layers.Conv2DTranspose(256,(20,20),strides=(20,20),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv2DTranspose(80,(3,3),strides=(2,2),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv2D(40,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    
    for kk in range(0,2):
        dx = keras.layers.Conv2D(80,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
        dx = keras.layers.Dropout(dropout)(dx)
        dx = keras.layers.Conv2D(40,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
        dx = keras.layers.Dropout(dropout)(dx)
        x  = keras.layers.Add()([x,dx])

    if include_covariates==False:
    # if no SST and SSS used as covariates
        x = keras.layers.Conv2D(x_train.shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    else:
    # else
        x = keras.layers.Conv2D(int(x_train.shape[3]/3),(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    decoder       = keras.models.Model(decoder_input,x)
      
    encoder.summary()
    decoder.summary()

    input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    x          = decoder(encoder([input_data,mask]))
    model_AE   = keras.models.Model([input_data,mask],x)
  
    #model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
    if include_covariates==False:
        size_tw = x_train.shape[3]
    else:
        size_tw = int(x_train.shape[3]/3)
    model_AE.compile(loss=keras_custom_loss_function(size_tw),optimizer=keras.optimizers.Adam(lr=1e-3))
    model_AE.summary()

    if DimCAE > x_train.shape[0] :
        DimCAE = DimAE

    return genFilename, encoder, decoder, model_AE, DimCAE

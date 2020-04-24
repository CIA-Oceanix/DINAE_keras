from dinae_keras import *

def flagProcess2_4(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

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
        x = keras.layers.Conv2D(256,(5,5),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
        x = keras.layers.Dropout(dropout)(x)
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

    return genFilename, encoder, decoder, model_AE, DimCAE

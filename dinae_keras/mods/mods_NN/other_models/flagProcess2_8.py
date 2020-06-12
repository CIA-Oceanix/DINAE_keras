from dinae_keras import *

def flagProcess2_8(dict_global_Params,genFilename,x_data,mask_data):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")


    input_layer = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    mask        = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
 
   
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
                 
    decoder_input = keras.layers.Input(shape=(int(np.floor(x_data.shape[1]/32)),int(np.floor(x_data.shape[2]/32)),DimAE))    
    
    x = keras.layers.Conv2DTranspose(64,(16,16),strides=(16,16),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv2DTranspose(32,(3,3),strides=(2,2),use_bias=False,activation='relu',padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv2D(16,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    
    x = keras.layers.Conv2D(x_data.shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    decoder       = keras.models.Model(decoder_input,x)
 
    encoder.summary()
    decoder.summary()
 
    input_data = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    mask       = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    x          = decoder(encoder([input_data,mask]))
    model_AE   = keras.models.Model([input_data,mask],x)
  
    model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
    model_AE.summary()
    DimCAE = DimAE * int(np.floor(x_data.shape[1]/32)) * int(np.floor(x_data.shape[2]/32))
    if DimCAE > x_data.shape[0] :
        DimCAE = DimAE

    return genFilename, encoder, decoder, model_AE, DimCAE

from dinae import *

def flagProcess2_5(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

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

    return genFilename, encoder, decoder, model_AE, DimCAE

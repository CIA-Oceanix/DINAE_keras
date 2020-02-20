from dinae import *

def flagProcess2_1(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")


    input_data = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
   
    x = keras.layers.Conv2D(6*DimAE,(x_train.shape[1],x_train.shape[2]),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(input_data)
    x = keras.layers.Conv2D(2*DimAE,(1,1),activation='relu', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
    x = keras.layers.Conv2D(DimAE,(1,1),activation='linear', padding='valid',kernel_regularizer=keras.regularizers.l2(wl2))(x)
    
    encoder    = keras.models.Model([input_data,mask],x)   
    decoder_input = keras.layers.Input(shape=(1,1,DimAE))
  
    x = keras.layers.Conv2DTranspose(2*DimAE,(x_train.shape[1],x_train.shape[2]),strides=(x_train.shape[1],x_train.shape[2]),padding='same',output_padding=None,kernel_regularizer=keras.regularizers.l2(wl2))(decoder_input)
    for kk in range(0,2):
      dx = keras.layers.Conv2D(5*DimAE,(3,3),activation='relu', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
      dx = keras.layers.Conv2D(2*DimAE,(3,3), padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(dx)
      x  = keras.layers.Add()([x,dx])
  
    x = keras.layers.Conv2D(x_train.shape[3],(1,1),activation='linear', padding='same',kernel_regularizer=keras.regularizers.l2(wl2))(x)
  
    decoder       = keras.models.Model(decoder_input,x)
  
    encoder.summary()
    decoder.summary()
  
    input_data = keras.layers.Input(shape=(x_train.shape[1]*x_train.shape[2]*x_train.shape[3],))
    mask       = keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    x          = decoder(encoder([input_data,mask]))
    model_AE     = keras.models.Model([input_data,mask],x)
   
    model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))
    model_AE.summary()

    return genFilename, encoder, decoder, model_AE, DimCAE

from dinae_keras import *

def flagProcess2_0(dict_global_Params,genFilename,x_data,mask_data):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    ## auto-encoder architecture (MLP)
    input_data = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    mask       = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
   
    x          = keras.layers.Flatten()(input_data)
  
    encodeda   = keras.layers.Dense(DimAE, activation='linear')(x)
    encoder    = keras.models.Model([input_data,mask],encodeda)
  
    decoder_input = keras.layers.Input(shape=(DimAE,))
    decodedb      = keras.layers.Dense(x_data.shape[1]*x_data.shape[2]*x_data.shape[3])(decoder_input)
    decodedb      = keras.layers.Reshape((x_data.shape[1],x_data.shape[2],x_data.shape[3]))(decodedb)
    decoder       = keras.models.Model(decoder_input,decodedb)
  
    encoder.summary()
    decoder.summary()
  
    input_data = keras.layers.Input(shape=(x_data.shape[1]*x_data.shape[2]*x_data.shape[3],))
    mask       = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
    x          = decoder(encoder([input_data,mask]))
    model_AE     = keras.models.Model([input_data,mask],x)
  
    model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-2))
    model_AE.summary()

    ## auto-encoder architecture (MLP)
    if 1*0:
        input_layer = keras.layers.Input(shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
      
        x          = keras.layers.Flatten()(input_layer)
      
        x          = keras.layers.Dense(6*DimAE, activation='relu')(x)
        x          = keras.layers.Dense(2*DimAE, activation='relu')(x)
        encodeda   = keras.layers.Dense(DimAE, activation='linear')(x)
        encoder    = keras.models.Model(input_layer,encodeda)
      
        decoder_input = keras.layers.Input(shape=(DimAE,))
        x     = keras.layers.Dense(10*DimAE, activation='relu')(decoder_input)
        x     = keras.layers.Dense(20*DimAE, activation='relu')(x)
        decodedb      = keras.layers.Dense(x_data.shape[1]*x_data.shape[2]*x_data.shape[3])(x)
        decodedb      = keras.layers.Reshape((x_data.shape[1],x_data.shape[2],x_data.shape[3]))(decodedb)
        decoder       = keras.models.Model(decoder_input,decodedb)
      
        encoder.summary()
        decoder.summary()
      
        input_data = keras.layers.Input(shape=(x_data.shape[1]*x_data.shape[2]*x_data.shape[3],))
        x          = decoder(encoder(input_layer))
        model_AE     = keras.models.Model(input_layer,x)
      
        model_AE.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-2))
        model_AE.summary()

    return genFilename, encoder, decoder, model_AE, DimCAE

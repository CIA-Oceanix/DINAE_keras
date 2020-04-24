from dinae_keras import *

def define_GradModel(model_AE,shape,flagGradModel=0):

    x_input         = keras.layers.Input((shape[1],shape[2],2*shape[3]))

    # Define gradMaskModel 
    x = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x_input)
    for nn in range(0,10):
        x = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    x = keras.layers.Conv2D(shape[3],(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
    gradMaskModel  = keras.models.Model([x_input],[x])

    # Define gradModel (convolutional model for gradient)
    
    # ResNet
    if flagGradModel == 0:
        print("...... Initialize Gradient Model: ResNet")
        x_input  = keras.layers.Input((shape[1],shape[2],2*shape[3]))
        x        = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x_input)
        x        = keras.layers.Conv2D(5,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
        for nn in range(0,10):
            dx = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
            dx = keras.layers.Conv2D(5,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
            x  = keras.layers.Add()([x,dx])
        x = keras.layers.Conv2D(shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
        gradModel  = keras.models.Model([x_input],[x])

    # ResNet with one-step memory
    elif flagGradModel == 1:
        print("...... Initialize Gradient Model: ResNet with one-step memory")
        x_input  = keras.layers.Input((shape[1],shape[2],shape[3]*2))
        x        = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x_input)
        x        = keras.layers.Conv2D(5,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
        for nn in range(0,10):
            dx = keras.layers.Conv2D(10,(3,3),activation='relu', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
            dx = keras.layers.Conv2D(5,(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(dx)
            x  = keras.layers.Add()([x,dx])
        x = keras.layers.Conv2D(shape[3],(3,3),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(x)
        gradModel  = keras.models.Model([x_input],[x])
      
    # LSTM
    elif flagGradModel == 2:
        print("...... Initialize Gradient Model: LSTM")
        x_input  = keras.layers.Input((shape[1],shape[2],shape[3]*2))
        gx_lstm  = keras.layers.ConvLSTM2D(10, kernel_size=(1,3), padding='same', use_bias=False, activation='relu')(keras.layers.Reshape((1,shape[1],shape[2],2))(x_input))#,return_state=True)#,input_shape=(1, para_Model.NbHS))
        gx_lstm  = keras.layers.Conv2D(shape[3],(1,1),activation='linear', padding='same',use_bias=False,kernel_regularizer=keras.regularizers.l2(wl2))(gx_lstm)
        gradModel  = keras.models.Model([x_input],[gx_lstm])
 
    # return gradModel and gradMaskModel
    gradMaskModel.summary()
    gradModel.summary()

    return gradModel,gradMaskModel
  

# launch.py

The top script, with global parameters used in the AE and main calls to the functions: data reading, training, evaluations, plots...

# dinae/mods/flagProcess0.py
Read the model, the observations, the mask and the OI

# dinae/mods/flagProcess1.py
Create missing data (based on the mask variables)

# dinae/mods/flagProcess2.py
Design of the autoencoder:

* mods_flagProcess2/flagProcess2_6.py:
  * Definition of the Encoder:   encoder = keras.models.Model([input_layer,mask],x)
    * input_layer (keras.layers.Input): 3-dimensional (training dataset shape)
    * mask (keras.layers.Input): 3-dimensional 
    * x: add all the layers (Conv2D+Pooling+Dropout)
  * Definition of the decoder:   decoder = keras.models.Model(decoder_input,x)
    * decoder_input: (keras.layers.Input) : 3-dimensional
    * x: add all the layer (Conv2D+Dropout)
  * Full definition of the model: model = keras.models.Model([input_data,mask],x)
    * input_data: 3-dimensional (training dataset shape) 
    * mask: 3-dimensional (training dataset shape) 
    * x: decoder(encoder([input_data,mask]))
  * Compile model: model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=1e-3))

mods_flagProcess2/flagProcess2_7.py:
mods_flagProcess2/flagProcess2_8.py:

# dinae/mods/flagProcess3.py
Define classifiers

# dinae/mods/flagProcess4.py
Train and evaluate AE

* mods_flagProcess4/eval_Performance.py:
  * eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt): compute the scores of AE reconstruction from training and test datasets
  * eval_InterpPerformance: compute the scores of the interpolation (where there is no data)

mods_flagProcess4/def_DINConvAE.py:
  *

mods_flagProcess4/def_GradModel.py:

mods_flagProcess4/def_GradDINConvAE.py:



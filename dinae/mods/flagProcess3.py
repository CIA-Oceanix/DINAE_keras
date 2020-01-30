from dinae import *

def flagProcess3(dict_global_Params,y_train):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    num_classes = (np.max(y_train)+1).astype(int)
    
    classifier = keras.Sequential()
    classifier.add(keras.layers.Dense(32,activation='relu', input_shape=(DimAE,)))
    classifier.add(keras.layers.Dense(64,activation='relu'))
    classifier.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    classifier.summary()

    return classifier


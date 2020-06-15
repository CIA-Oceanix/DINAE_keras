from dinae_keras import *
from .mods_DIN.def_GradModel         import define_GradModel
from .mods_DIN.def_GradDINConvAE     import define_GradDINConvAE

def load_Models_GB(dict_global_Params, genFilename, shape, fileModels, encoder, decoder, model_AE, params):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    DimCAE = DimAE

    nProjInit, nGradInit, lrInit = params
    print('.................. Load Encoder/Decoder ')
    encoder.load_weights(fileModels[0])
    decoder.load_weights(fileModels[1].replace('Encoder','Decoder'))

    for layer in encoder.layers:
        layer.trainable = True
    for layer in decoder.layers:
        layer.trainable = True

    print("..... Initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(nProjInit,nGradInit))
    gradModel,gradMaskModel =  define_GradModel(shape,flagGradModel,wl2)
    gradMaskModel.load_weights(fileModels[0].replace('Encoder','GradMaskModel'))
    gradModel.load_weights(fileModels[0].replace('Encoder','GradModel'))

    global_model_Grad,global_model_Grad_Masked = define_GradDINConvAE(nProjInit,nGradInit,\
                                                  model_AE,shape,gradModel,gradMaskModel,flagGradModel,\
                                                  flagUseMaskinEncoder,size_tw,include_covariates,N_cov)
    if flagTrOuputWOMissingData == 1:
        global_model_Grad.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrInit))
    else:
        global_model_Grad_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrInit))

    return gradModel, gradMaskModel, global_model_Grad, global_model_Grad_Masked

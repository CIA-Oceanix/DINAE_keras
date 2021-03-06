from dinae_keras import *
from .mods_DIN.def_DINConvAE         import define_DINConvAE

def load_Models_FP(dict_global_Params, genFilename, shape, fileModels, encoder, decoder, model_AE,  params):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    DimCAE = DimAE

    nProjInit, lrInit = params 
    print('.................. Load Encoder/Decoder ')
    encoder.load_weights(fileModels[0])
    decoder.load_weights(fileModels[1])

    print("..... Initialize number of projections in DINCOnvAE model # %d"%(nProjInit))
    global_model_FP,global_model_FP_Masked = define_DINConvAE(nProjInit,model_AE,\
                                                              shape,flag_MultiScaleAEModel,\
                                                              flagUseMaskinEncoder,size_tw,include_covariates,N_cov)
    if flagTrOuputWOMissingData == 1:
        global_model_FP.compile(loss='mean_squared_error',\
                                optimizer=keras.optimizers.Adam(lr=lrInit))
    else:
        global_model_FP_Masked.compile(loss='mean_squared_error',\
                               optimizer=keras.optimizers.Adam(lr=lrInit))

 
    return global_model_FP, global_model_FP_Masked

from dinae_keras import *
from .mods_NN.ConvAE import ConvAE 
from .mods_NN.GENN   import GENN 

def define_Models(dict_global_Params,genFilename,x,mask):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")
    if flagAEType == 1: ## Conv-AE for SST case-study §64x64)  
      genFilename, encoder, decoder, model_AE, DimCAE = ConvAE(dict_global_Params,genFilename,x,mask)
    if flagAEType == 2: ## Energy function of the type ||x(p)-f(x(q, q<>p))||
      genFilename, encoder, decoder, model_AE, DimCAE = GENN(dict_global_Params,1,0,genFilename,x,mask)
    return genFilename, encoder, decoder, model_AE, DimCAE

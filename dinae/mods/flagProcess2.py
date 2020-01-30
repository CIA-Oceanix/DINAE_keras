from dinae import *
from .flagProcess2_0 import flagProcess2_0 as fl20
from .flagProcess2_1 import flagProcess2_1 as fl21
from .flagProcess2_2 import flagProcess2_2 as fl22
from .flagProcess2_3 import flagProcess2_3 as fl23
from .flagProcess2_4 import flagProcess2_4 as fl24
from .flagProcess2_5 import flagProcess2_5 as fl25
from .flagProcess2_6 import flagProcess2_6 as fl26
from .flagProcess2_7 import flagProcess2_7 as fl27
from .flagProcess2_8 import flagProcess2_8 as fl28

def flagProcess2(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    DimCAE = DimAE
    
    if flagAEType == 0: ## MLP-AE
      genFilename, encoder, decoder, model_AE, DimCAE = fl20(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test)
    elif flagAEType == 1: ## Conv-AE
      genFilename, encoder, decoder, model_AE, DimCAE = fl21(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test)
    elif flagAEType == 2: ## Conv-AE
      genFilename, encoder, decoder, model_AE, DimCAE = fl22(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test) 
    elif flagAEType == 3: ## Conv-AE for SST case-study
      genFilename, encoder, decoder, model_AE, DimCAE = fl23(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test)  
    elif flagAEType == 4: ## Conv-AE for SST case-study §64x64)  
      genFilename, encoder, decoder, model_AE, DimCAE = fl24(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test)
    elif flagAEType == 5: ## Conv-AE for SST case-study
      genFilename, encoder, decoder, model_AE, DimCAE = fl25(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test)  
    elif flagAEType == 6: ## Conv-AE for SST case-study §64x64)  
      genFilename, encoder, decoder, model_AE, DimCAE = fl26(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test)     
    elif flagAEType == 7: ## Energy function of the type ||x(p)-f(x(q, q<>p))||
      genFilename, encoder, decoder, model_AE, DimCAE = fl27(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test)
    elif flagAEType == 8: ## Conv-AE for SST case-study §64x64)  
      genFilename, encoder, decoder, model_AE, DimCAE = fl28(dict_global_Params,genFilename,x_train,mask_train,x_test,mask_test)

    return genFilename, encoder, decoder, model_AE, DimCAE

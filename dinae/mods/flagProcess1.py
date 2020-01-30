from dinae import *

def flagProcess1(dict_global_Params,x_train,mask_train,x_test,mask_test):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    print("..... Use missing data masks from file ")
    Nsquare = 0
    x_train_missing = x_train * mask_train
    x_test_missing  = x_test  * mask_test

    return x_train_missing, x_test_missing

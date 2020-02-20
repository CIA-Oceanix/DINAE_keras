from dinae import *

def flagProcess1(dict_global_Params,x_train,mask_train,err_train,x_test,mask_test,err_test):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    print("..... Use missing data masks from file ")
    gt_train = x_train
    x_train_missing = x_train * mask_train + err_train
    gt_test  = x_test 
    x_test_missing  = x_test  * mask_test  + err_test

    return gt_train, x_train_missing, gt_test, x_test_missing

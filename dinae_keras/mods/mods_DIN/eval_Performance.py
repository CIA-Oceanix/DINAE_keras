from dinae_keras import *

# functions for the evaluation of interpolation and auto-encoding performance
def eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt):

    mse_AE_Tr        = np.mean( (rec_AE_Tr - x_train)**2 )
    var_Tr           = np.mean( (x_train-np.mean(x_train,axis=0)) ** 2 )
    exp_var_AE_Tr    = 1. - mse_AE_Tr / var_Tr
    
    mse_AE_Tt        = np.mean( (rec_AE_Tt - x_test)**2 )
    var_Tt           = np.mean( (x_test-np.mean(x_train,axis=0))** 2 )
    exp_var_AE_Tt    = 1. - mse_AE_Tt / var_Tt
            
    return exp_var_AE_Tr,exp_var_AE_Tt

def eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,
                           mask_test,x_test,x_test_missing,x_test_pred):
    mse_train      = np.zeros((2))
    mse_train[0]   = np.sum( mask_train * (x_train_pred - x_train_missing)**2 ) / np.sum( mask_train )
    mse_train[1]   = np.mean( (x_train_pred - x_train)**2 )
    exp_var_train  = 1. - mse_train #/ var_Tr
            
    mse_test        = np.zeros((2))
    mse_test[0]     = np.sum( mask_test * (x_test_pred - x_test_missing)**2 ) / np.sum( mask_test )
    mse_test[1]     = np.mean( (x_test_pred - x_test)**2 ) 
    exp_var_test = 1. - mse_test #/ var_Tt

    mse_train_interp        = np.sum( (1.-mask_train) * (x_train_pred - x_train)**2 ) / np.sum( 1. - mask_train )
    exp_var_train_interp    = 1. - mse_train_interp 
    
    mse_test_interp        = np.sum( (1.-mask_test) * (x_test_pred - x_test)**2 ) / np.sum( 1. - mask_test )
    exp_var_test_interp    = 1. - mse_test_interp
            
    return mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp

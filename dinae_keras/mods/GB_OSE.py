from dinae_keras import *
from .tools import *
from .graphics import *
from .load_Models_GB                 import load_Models_GB
from .mods_DIN.eval_Performance      import eval_AEPerformance
from .mods_DIN.eval_Performance      import eval_InterpPerformance
from .mods_DIN.def_GradModel         import define_GradModel
from .mods_DIN.def_GradDINConvAE     import define_GradDINConvAE
from .mods_DIN.plot_Figs             import plot_Figs_Tt
from .mods_DIN.save_Models           import save_Models

def GB_OSE(dict_global_Params,genFilename,\
                        meanTt,stdTt,x_test,x_test_missing,mask_test,gt_test,lday_test,x_train_OI,x_test_OI,\
                        encoder,decoder,model_AE,DimCAE):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    ## Load models
    if domain=="GULFSTREAM":
        weights_Encoder="/gpfsscratch/rech/yrf/uba22to/DINAE/GULFSTREAM"+\
                        "/resIA_nadir_nadlag_5_obs/GB_GENN_wwmissing_wOI/"+\
                        "modelNATL60_SSH_275_200_200_dW000WFilter011_NFilter200_"+\
                        "RU010_LR004woSR_Alpha100_AE07D200N03W04_Nproj05_Encoder_iter019.mod"
        weights_Decoder="/gpfsscratch/rech/yrf/uba22to/DINAE/GULFSTREAM"+\
                        "/resIA_nadir_nadlag_5_obs/GB_GENN_wwmissing_wOI/"+\
                        "modelNATL60_SSH_275_200_200_dW000WFilter011_NFilter200_"+\
                        "RU010_LR004woSR_Alpha100_AE07D200N03W04_Nproj05_Decoder_iter019.mod"
    elif domain=="OSMOSIS":
        weights_Encoder="/gpfsscratch/rech/yrf/uba22to/DINAE/OSMOSIS"+\
                        "/resIA_nadir_nadlag_5_obs/GB_GENN_wwmissing_wOI/"+\
                        "modelNATL60_SSH_275_200_200_dW000WFilter011_NFilter200_"+\
                        "RU010_LR004woSR_Alpha100_AE07D200N03W04_Nproj05_Encoder_iter019.mod"
        weights_Decoder="/gpfsscratch/rech/yrf/uba22to/DINAE/OSMOSIS"+\
                        "/resIA_nadir_nadlag_5_obs/GB_GENN_wwmissing_wOI/"+\
                        "modelNATL60_SSH_275_200_200_dW000WFilter011_NFilter200_"+\
                        "RU010_LR004woSR_Alpha100_AE07D200N03W04_Nproj05_Decoder_iter019.mod"
    global_model_Grad, global_model_Grad_Masked =\
        load_Models_GB(dict_global_Params, genFilename, x_test.shape,fileModels,\
                       encoder,decoder,model_AE,[2,1e-3])

    ## initialization
    x_test_init  = np.copy(x_test_missing)

    # *********************** #
    # Prediction on test data #
    # *********************** #

    # trained full-model
    x_test_pred     = global_model_GB.predict([x_test_init,mask_test])

    # trained AE applied to gap-free data
    if flagUseMaskinEncoder == 1:
        rec_AE_Tt     = model_AE.predict([x_test,np.zeros((mask_train.shape))])
    else:
        rec_AE_Tt     = model_AE.predict([x_test,np.ones((mask_test.shape))])

    # remove additional covariates from variables
    if include_covariates == True:
        mask_test_wc, x_test_wc, x_test_init_wc, x_test_missing_wc=\
        mask_test, x_test, x_test_init, x_test_missing
        index = np.arange(0,(N_cov+1)*size_tw,(N_cov+1))
        mask_test      = mask_test[:,:,:,index]
        x_test         = x_test[:,:,:,index]
        x_test_init    = x_test_init[:,:,:,index]
        x_test_missing = x_test_missing[:,:,:,index]
        meanTt, stdTt  = meanTt[0], stdTt[0]
        
    idT = int(np.floor(x_test.shape[3]/2))
    saved_path = dirSAVE+'/saved_path_GB_'+suf1+'_'+suf2+'.pickle'
    if flagloadOIData == 1:
        # generate some plots
        plot_Figs_Tt(dirSAVE,domain,genFilename,genSuffixModel,\
                     (gt_test*stdTt)+meanTt+x_test_OI,(x_test_missing*stdTt)+meanTt+x_test_OI,mask_test,lday_test,\
                     (x_test_pred*stdTt)+meanTt+x_test_OI,(rec_AE_Tt*stdTt)+meanTt+x_test_OI)
        # Save DINAE result         
        with open(saved_path, 'wb') as handle:
            pickle.dump([((gt_test*stdTt)+meanTt+x_test_OI)[:,:,:,idT],((x_test_missing*stdTt)+meanTt+x_test_OI)[:,:,:,idT],\
                         ((x_test_pred*stdTt)+meanTt+x_test_OI)[:,:,:,idT],((rec_AE_Tt*stdTt)+meanTt+x_test_OI)[:,:,:,idT]], handle)

    else:
        # generate some plots
        plot_Figs_Tt(dirSAVE,domain,genFilename,genSuffixModel,\
                     (gt_test*stdTt)+meanTt,(x_test_missing*stdTt)+meanTt,mask_test,lday_test,\
                     (x_test_pred*stdTt)+meanTt,(rec_AE_Tt*stdTt)+meanTt)
        # Save DINAE result         
        with open(saved_path, 'wb') as handle:
            pickle.dump([((gt_test*stdTt)+meanTt)[:,:,:,idT],((x_test_missing*stdTt)+meanTt)[:,:,:,idT],\
                         ((x_test_pred*stdTt)+meanTt)[:,:,:,idT],((rec_AE_Tt*stdTt)+meanTt)[:,:,:,idT]], handle)


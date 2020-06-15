from dinae_keras import *
from .tools import *
from .graphics import *
from .load_Models_FP                 import load_Models_FP
from .mods_DIN.eval_Performance      import eval_AEPerformance
from .mods_DIN.eval_Performance      import eval_InterpPerformance
from .mods_DIN.def_DINConvAE         import define_DINConvAE
from .mods_DIN.plot_Figs             import plot_Figs_Tt
from .mods_DIN.save_Models           import save_Models

def FP_OSE(dict_global_Params,genFilename,\
                        meanTt,stdTt,x_test,x_test_missing,mask_test,gt_test,lday_test,x_train_OI,x_test_OI,\
                        encoder,decoder,model_AE,DimCAE):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    x_test_init  = np.nan_to_num(np.copy(x_test_missing))

    if load_Model==True:
        ## Load models
        if domain=="GULFSTREAM":
            weights_Encoder="/gpfsscratch/rech/yrf/uba22to/DINAE/GULFSTREAM"+\
                            "/resIA_nadir_nadlag_5_obs/FP_GENN_wwmissing_wOI/"+\
                            "modelNATL60_SSH_275_200_200_dW000WFilter011_NFilter200_"+\
                            "RU010_LR004woSR_Alpha100_AE07D200N03W04_Nproj05_Encoder_iter019.mod"
        elif domain=="OSMOSIS":
            weights_Encoder="/gpfsscratch/rech/yrf/uba22to/DINAE/OSMOSIS"+\
                            "/resIA_nadir_nadlag_5_obs/FP_GENN_wwmissing_wOI/"+\
                            "modelNATL60_SSH_275_200_200_dW000WFilter011_NFilter200_"+\
                            "RU010_LR004woSR_Alpha100_AE07D200N03W04_Nproj05_Encoder_iter019.mod"
        weights_Decoder=weights_Encoder.replace('Encoder','Decoder')
        global_model_FP, global_model_FP_Masked = load_Models_FP(dict_global_Params,\
                                                  genFilename, x_test.shape,[weights_Encoder,weights_Decoder],\
                                                  encoder,decoder,model_AE,[5,1e-4])
    else:
        ## Train models
        NbProjection   = [5,5,5,5]
        lrUpdate   = [1e-4,1e-5,1e-6,1e-7]
        IterUpdate     = [0,3,10,15,20,25,30,35,40]
        val_split      = 0.1
        comptUpdate    = 0
        print("..... Start learning AE model %d FP/Grad %s"%(flagAEType,flagOptimMethod))
        for iter in range(0,Niter):
            if iter == IterUpdate[comptUpdate]:
                # update DINConvAE model
                NBProjCurrent = NbProjection[comptUpdate]
                print("..... Update/initialize number of projections in DINCOnvAE model # %d"%(NbProjection[comptUpdate]))
                global_model_FP,global_model_FP_Masked = define_DINConvAE(NbProjection[comptUpdate],model_AE,x_test.shape,\
                                                                          flag_MultiScaleAEModel,flagUseMaskinEncoder,\
                                                                          size_tw,include_covariates,N_cov)
                global_model_FP_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                if comptUpdate < len(NbProjection)-1:
                    comptUpdate += 1

            # FP-based iteration            
            history = global_model_FP_Masked.fit([x_test_init,mask_test],[np.zeros((x_test_init.shape[0],1))],
                                                    batch_size=batch_size,
                                                    epochs = NbEpoc,
                                                    verbose = 1,
                                                    validation_split=val_split)
            genSuffixModel=save_Models(dict_global_Params,genFilename,NBProjCurrent,encoder,decoder,iter)

    # *********************** #
    # Prediction on test data #
    # *********************** #

    # trained full-model
    x_test_pred     = global_model_FP.predict([x_test_init,mask_test])
    weights = global_model_FP.get_weights()

    # trained AE applied to gap-free data
    if flagUseMaskinEncoder == 1:
        rec_AE_Tt     = model_AE.predict([x_test,np.zeros((mask_test.shape))])
    else:
        rec_AE_Tt     = model_AE.predict([x_test,np.ones((mask_test.shape))])

    # remove additional covariates from variables
    if include_covariates == True:
        mask_test_wc, x_test_wc, x_test_missing_wc=\
        mask_test, x_test, x_test_missing
        index = np.arange(0,(N_cov+1)*size_tw,(N_cov+1))
        mask_test      = mask_test[:,:,:,index]
        x_test         = x_test[:,:,:,index]
        x_test_missing = x_test_missing[:,:,:,index]
        meanTt, stdTt  = meanTt[0], stdTt[0]
 
    idT = int(np.floor(x_test.shape[3]/2)) 
    saved_path = dirSAVE+'/saved_path_FP_'+suf1+'_'+suf2+'.pickle'

    alpha=[1.,0.,0.]
    NBProjCurrent=5
    NBGradCurrent=0
    genSuffixModel = '_Alpha%03d'%(100*alpha[0]+10*alpha[1]+alpha[2])
    if flagUseMaskinEncoder == 1:
        genSuffixModel = genSuffixModel+'_MaskInEnc'
        if stdMask  > 0:
            genSuffixModel = genSuffixModel+'_Std%03d'%(100*stdMask)
    if flagTrOuputWOMissingData == 1:
        genSuffixModel = genSuffixModel+'_AETRwoMissingData'+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))
    else:
        genSuffixModel = genSuffixModel+'_AE'+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))
    if flagloadOIData == 1:
        # generate some plots
        plot_Figs_Tt(dirSAVE,domain,genFilename,genSuffixModel,\
                     (gt_test*stdTt)+meanTt+x_test_OI,(x_test_missing*stdTt)+meanTt+x_test_OI,mask_test,lday_test,\
                     (x_test_pred*stdTt)+meanTt+x_test_OI,(rec_AE_Tt*stdTt)+meanTt+x_test_OI)
        # Save DINAE result         
        with open(saved_path, 'wb') as handle:
            pickle.dump([((gt_test*stdTt)+meanTt+x_test_OI)[:,:,:,idT],x_test_OI[:,:,:,idT],((x_test_missing*stdTt)+meanTt+x_test_OI)[:,:,:,idT],\
                         ((x_test_pred*stdTt)+meanTt+x_test_OI)[:,:,:,idT],((rec_AE_Tt*stdTt)+meanTt+x_test_OI)[:,:,:,idT]], handle)

    else:
        # generate some plots
        plot_Figs_Tt(dirSAVE,domain,genFilename,genSuffixModel,\
                     (gt_test*stdTt)+meanTt,(x_test_missing*stdTt)+meanTt,mask_test,lday_test,\
                     (x_test_pred*stdTt)+meanTt,(rec_AE_Tt*stdTt)+meanTt,)
        # Save DINAE result         
        with open(saved_path, 'wb') as handle:
            pickle.dump([((gt_test*stdTt)+meanTt)[:,:,:,idT],x_test_OI[:,:,:,idT],((x_test_missing*stdTt)+meanTt)[:,:,:,idT],\
                         ((x_test_pred*stdTt)+meanTt)[:,:,:,idT],((rec_AE_Tt*stdTt)+meanTt)[:,:,:,idT]], handle)


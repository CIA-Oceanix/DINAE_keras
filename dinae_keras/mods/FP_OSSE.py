from dinae_keras import *
from .tools import *
from .graphics import *
from .load_Models_FP                 import load_Models_FP
from .mods_DIN.eval_Performance      import eval_AEPerformance
from .mods_DIN.eval_Performance      import eval_InterpPerformance
from .mods_DIN.def_DINConvAE         import define_DINConvAE
from .mods_DIN.def_GradModel         import define_GradModel
from .mods_DIN.def_GradDINConvAE     import define_GradDINConvAE
from .mods_DIN.plot_Figs             import plot_Figs
from .mods_DIN.save_Models           import save_Models

def FP_OSSE(dict_global_Params,genFilename,x_train,x_train_missing,mask_train,gt_train,\
                        meanTr,stdTr,x_test,x_test_missing,mask_test,gt_test,lday_test,x_train_OI,x_test_OI,encoder,decoder,model_AE,DimCAE):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    # ******************************** #
    # PCA decomposition for comparison #
    # *********************************#

    # train PCA
    pca      = decomposition.PCA(DimCAE)
    pca.fit(np.reshape(gt_train,(gt_train.shape[0],gt_train.shape[1]*gt_train.shape[2]*gt_train.shape[3])))
    
    # apply PCA to test data
    rec_PCA_Tt       = pca.transform(np.reshape(gt_test,(gt_test.shape[0],gt_test.shape[1]*gt_test.shape[2]*gt_test.shape[3])))
    rec_PCA_Tt[:,DimCAE:] = 0.
    rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
    mse_PCA_Tt       = np.mean( (rec_PCA_Tt - gt_test.reshape((gt_test.shape[0],gt_test.shape[1]*gt_test.shape[2]*gt_test.shape[3])))**2 )
    var_Tt           = np.mean( (gt_test-np.mean(gt_train,axis=0))** 2 )
    exp_var_PCA_Tt   = 1. - mse_PCA_Tt / var_Tt
    
    print(".......... PCA Dim = %d"%(DimCAE))
    print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1]))
    print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))

    print("..... Regularization parameters: dropout = %.3f, wl2 = %.2E"%(dropout,wl2))
    
    # ***************** #
    # model compilation #
    # ***************** #

    # model fit
    NbProjection   = [0,0,2,2,5,5,10,15,14]
    NbProjection   = [5,5,5,5]
    lrUpdate       = [1e-3,1e-4,1e-5,1e-5,1e-5,1e-6,1e-6,1e-5,1e-6]
    if flagTrOuputWOMissingData==0:
        lrUpdate   = [1e-4,1e-5,1e-6,1e-7]
    else:
        lrUpdate   = [1e-3,1e-4,1e-5,1e-6]
    IterUpdate     = [0,3,10,15,20,25,30,35,40]
    #IterUpdate     = [0,6,15,20]
    val_split      = 0.1
    
    iterInit = 0
    IterTrainAE = 0
    IterUpdateInit = 10000
    
    ## initialization
    x_train_init = np.copy(x_train_missing)
    x_test_init  = np.copy(x_test_missing)

    comptUpdate = 0
    if flagLoadModel == 1:
        global_model_FP, global_model_FP_Masked =\
        load_Models_FP(dict_global_Params, genFilename, x_train.shape,fileAEModelInit,[2,1e-3])

    # ******************** #
    # Start Learning model #
    # ******************** #
        
    print("..... Start learning AE model %d FP/Grad %s"%(flagAEType,flagOptimMethod))
    for iter in range(iterInit,Niter):
        if iter == IterUpdate[comptUpdate]:
            # update DINConvAE model
            NBProjCurrent = NbProjection[comptUpdate]
            print("..... Update/initialize number of projections in DINCOnvAE model # %d"%(NbProjection[comptUpdate]))
            global_model_FP,global_model_FP_Masked = define_DINConvAE(NbProjection[comptUpdate],model_AE,x_train.shape,\
                                                                          flag_MultiScaleAEModel,flagUseMaskinEncoder,\
                                                                          size_tw,include_covariates,N_cov)
            if flagTrOuputWOMissingData == 1:
                #global_model_FP.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                global_model_FP.compile(loss=keras_custom_loss_function(size_tw),optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
            else:
                global_model_FP_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
                #global_model_FP_Masked.compile(loss=keras_custom_loss_function(size_tw),optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
            if comptUpdate < len(NbProjection)-1:
                comptUpdate += 1
        
        # gradient descent iteration            
        if flagTrOuputWOMissingData == 1:
            history = global_model_FP.fit([x_train_init,mask_train],gt_train,
                  batch_size=batch_size,
                  epochs = NbEpoc,
                  verbose = 1, 
                  validation_split=val_split)
        else:
            history = global_model_FP_Masked.fit([x_train_init,mask_train],[np.zeros((x_train_init.shape[0],1))],
                  batch_size=batch_size,
                  epochs = NbEpoc,
                  verbose = 1, 
                  validation_split=val_split)

        # *********************** #
        # Prediction on test data #
        # *********************** #

        # trained full-model
        x_train_pred    = global_model_FP.predict([x_train_init,mask_train])
        x_test_pred     = global_model_FP.predict([x_test_init,mask_test])

        # trained AE applied to gap-free data
        if flagUseMaskinEncoder == 1:
            rec_AE_Tr     = model_AE.predict([x_train,np.zeros((mask_train.shape))])
            rec_AE_Tt     = model_AE.predict([x_test,np.zeros((mask_train.shape))])
        else:
            rec_AE_Tr     = model_AE.predict([x_train,np.ones((mask_train.shape))])
            rec_AE_Tt     = model_AE.predict([x_test,np.ones((mask_test.shape))])

        # remove additional covariates from variables
        if include_covariates == True:
            mask_train_wc, x_train_wc, x_train_init_wc, x_train_missing_wc,\
            mask_test_wc, x_test_wc, x_test_init_wc, x_test_missing_wc,\
            meanTr_wc, stdTr_wc=\
            mask_train, x_train, x_train_init, x_train_missing,\
            mask_test, x_test, x_test_init, x_test_missing,\
            meanTr, stdTr
            index = np.arange(0,(N_cov+1)*size_tw,(N_cov+1))
            mask_train      = mask_train[:,:,:,index]
            x_train         = x_train[:,:,:,index]
            x_train_init    = x_train_init[:,:,:,index]
            x_train_missing = x_train_missing[:,:,:,index]
            mask_test      = mask_test[:,:,:,index]
            x_test         = x_test[:,:,:,index]
            x_test_init    = x_test_init[:,:,:,index]
            x_test_missing = x_test_missing[:,:,:,index]
            meanTr = meanTr[0]
            stdTr  = stdTr[0]

        mse_train,exp_var_train,\
        mse_test,exp_var_test,\
        mse_train_interp,exp_var_train_interp,\
        mse_test_interp,exp_var_test_interp =\
        eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,\
                               mask_test,x_test,x_test_missing,x_test_pred)
        
        print(".......... iter %d"%(iter))
        print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
        print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
        print('....')
        print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
        print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
        print('....')
        print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
        print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))

        # interpolation and reconstruction score for the center image
        # when dealing with time series
        if x_train_init.shape[3] > 0 :
            
            dWCenter    = 32  
            
            dT = np.floor( x_train_init.shape[3] / 2 ).astype(int)
            mse_train_center        = np.mean( (x_train_pred[:,:,:,dT] - x_train[:,:,:,dT] )**2 )
            mse_train_center_interp = np.sum( (x_train_pred[:,:,:,dT]  - x_train[:,:,:,dT] )**2 * (1.-mask_train[:,:,:,dT])  ) / np.sum( (1.-mask_train[:,:,:,dT]) )
            
            mse_test_center         = np.mean( (x_test_pred[:,:,:,dT] - x_test[:,:,:,dT] )**2 )
            mse_test_center_interp  = np.sum( (x_test_pred[:,:,:,dT]  - x_test[:,:,:,dT] )**2 * (1.-mask_test[:,:,:,dT])  ) / np.sum( (1-mask_test[:,:,:,dT]) )
            
            var_train_center        = np.var(  x_train[:,:,:,dT] )
            var_test_center         = np.var(  x_test[:,:,:,dT] )
            
            exp_var_train_center         = 1.0 - mse_train_center / var_train_center
            exp_var_train_interp_center  = 1.0 - mse_train_center_interp / var_train_center
            exp_var_test_center          = 1.0 - mse_test_center  / var_test_center
            exp_var_test_interp_center   = 1.0 - mse_test_center_interp/ var_test_center
        print('.... Performance for "center" image')
        print('.... Image center variance (Tr)  : %.2f'%var_train_center)
        print('.... Image center variance (Tt)  : %.2f'%var_test_center)
        print('.... Error for all data (Tr)     : %.2e %.2f%%'%(mse_train_center*stdTr**2,100.*exp_var_train_center))
        print('.... Error for all data (Tt)     : %.2e %.2f%%'%(mse_test_center*stdTr**2,100.*exp_var_test_center))
        print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_center_interp*stdTr**2,100.*exp_var_train_interp_center))
        print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_center_interp*stdTr**2 ,100.*exp_var_test_interp_center))            
        print('   ')
        
        # AE performance of the trained AE applied to gap-free data
        exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
        
        print(".......... Auto-encoder performance when applied to gap-free data")
        print('.... explained variance AE (Tr)  : %.2f%%'%(100.*exp_var_AE_Tr))
        print('.... explained variance AE (Tt)  : %.2f%%'%(100.*exp_var_AE_Tt))
        
        if flagUseMaskinEncoder == 1:
        
            exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
        
            print('.... explained variance AE (Tr) with mask  : %.2f%%'%(100.*exp_var_AE_Tr))
            print('.... explained variance AE (Tt) with mask  : %.2f%%'%(100.*exp_var_AE_Tt))

        print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1]))
        print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))  

        # save models
        genSuffixModel=save_Models(dict_global_Params,genFilename,NBProjCurrent,encoder,decoder,iter)
 
        idT = int(np.floor(x_test.shape[3]/2))
        saved_path = dirSAVE+'/saved_path_%03d'%(iter)+'_FP_'+suf1+'_'+suf2+'.pickle'
        if flagloadOIData == 1:
            # generate some plots
            plot_Figs(dirSAVE,domain,genFilename,genSuffixModel,\
                  (gt_train*stdTr)+meanTr+x_train_OI,(x_train_missing*stdTr)+meanTr+x_train_OI,mask_train,\
                  (x_train_pred*stdTr)+meanTr+x_train_OI,(rec_AE_Tr*stdTr)+meanTr+x_train_OI,\
                  (gt_test*stdTr)+meanTr+x_test_OI,(x_test_missing*stdTr)+meanTr+x_test_OI,mask_test,lday_test,\
                  (x_test_pred*stdTr)+meanTr+x_test_OI,(rec_AE_Tt*stdTr)+meanTr+x_test_OI,iter)
            # Save DINAE result         
            with open(saved_path, 'wb') as handle:
                pickle.dump([((gt_test*stdTr)+meanTr+x_test_OI)[:,:,:,idT],((x_test_missing*stdTr)+meanTr+x_test_OI)[:,:,:,idT],\
                         ((x_test_pred*stdTr)+meanTr+x_test_OI)[:,:,:,idT],((rec_AE_Tt*stdTr)+meanTr+x_test_OI)[:,:,:,idT]], handle)

        else:
            # generate some plots
            plot_Figs(dirSAVE,domain,genFilename,genSuffixModel,\
                  (gt_train*stdTr)+meanTr,(x_train_missing*stdTr)+meanTr,mask_train,\
                  (x_train_pred*stdTr)+meanTr,(rec_AE_Tr*stdTr)+meanTr,\
                  (gt_test*stdTr)+meanTr,(x_test_missing*stdTr)+meanTr,mask_test,lday_test,\
                  (x_test_pred*stdTr)+meanTr,(rec_AE_Tt*stdTr)+meanTr,iter)
            # Save DINAE result         
            with open(saved_path, 'wb') as handle:
                pickle.dump([((gt_test*stdTr)+meanTr)[:,:,:,idT],((x_test_missing*stdTr)+meanTr)[:,:,:,idT],\
                         ((x_test_pred*stdTr)+meanTr)[:,:,:,idT],((rec_AE_Tt*stdTr)+meanTr)[:,:,:,idT]], handle)

        # reset variables with additional covariates
        if include_covariates == True:
            mask_train, x_train, x_train_init, x_train_missing,\
            mask_test, x_test, x_test_init, x_test_missing,\
            meanTr, stdTr=\
            mask_train_wc, x_train_wc, x_train_init_wc, x_train_missing_wc,\
            mask_test_wc, x_test_wc, x_test_init_wc, x_test_missing_wc,\
            meanTr_wc, stdTr_wc

        # update training data
        if iter > IterUpdateInit:
            # mask = 0(missing data) ; 1(data)
            if include_covariates == False: 
                x_train_init = mask_train * x_train_missing + (1.-mask_train) * x_train_pred
                x_test_init  = mask_test  * x_test_missing  + (1.-mask_test)  * x_test_pred
            else:
                index = np.arange(0,(N_cov+1)*size_tw,(N_cov+1))
                x_train_init[:,:,:,index] = mask_train[:,:,:,index] * x_train_missing[:,:,:,index] +\
                                            (1.-mask_train[:,:,:,index]) * x_train_pred
                x_test_init[:,:,:,index]  = mask_test[:,:,:,index]  * x_test_missing[:,:,:,index]  +\
                                            (1.-mask_test[:,:,:,index])  * x_test_pred

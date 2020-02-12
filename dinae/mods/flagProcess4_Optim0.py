from dinae import *
from .tools import *
from .graphics import *
from .mods_flagProcess4.eval_Performance      import eval_AEPerformance
from .mods_flagProcess4.eval_Performance      import eval_InterpPerformance
from .mods_flagProcess4.def_DINConvAE         import define_DINConvAE
from .mods_flagProcess4.def_GradModel         import define_GradModel
from .mods_flagProcess4.def_GradDINConvAE     import define_GradDINConvAE
from .mods_flagProcess4.plot_Figs             import plot_Figs
from .mods_flagProcess4.save_Models           import save_Models

def flagProcess4_Optim0(dict_global_Params,genFilename,x_train,x_train_missing,mask_train,meanTr,stdTr,x_test,x_test_missing,mask_test,lday_test,encoder,decoder,model_AE,DimCAE):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    # ******************************** #
    # PCA decomposition for comparison #
    # *********************************#

    # train PCA
    pca      = decomposition.PCA(DimCAE)
    pca.fit(np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])))
    
    # apply PCA to test data
    rec_PCA_Tt       = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
    rec_PCA_Tt[:,DimCAE:] = 0.
    rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
    mse_PCA_Tt       = np.mean( (rec_PCA_Tt - x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))**2 )
    var_Tt           = np.mean( (x_test-np.mean(x_train,axis=0))** 2 )
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
    lrUpdate       = [1e-3,1e-4,1e-3,1e-4,1e-5,1e-6,1e-6,1e-5,1e-6]
    IterUpdate     = [0,3,10,15,20,25,30,35,40]
    val_split      = 0.1
    
    flagLoadModelAE = 0
    # Here, specify a preloaded AE model
    fileAEModelInit = dirSAVE+'???.mod'
    
    iterInit = 0
    if flagLoadModelAE > 0 :
        iterInit = 13
    IterTrainAE = 0
    IterUpdateInit = 10000
    
    ## initialization
    x_train_init = np.copy(x_train_missing)
    x_test_init  = np.copy(x_test_missing)

    comptUpdate = 0
    if flagLoadModelAE > 0 :
        print('.................. Load Encoder/Decoder '+fileAEModelInit)
        encoder.load_weights(fileAEModelInit)
        decoder.load_weights(fileAEModelInit.replace('Encoder','Decoder'))

        comptUpdate = 3
        NBProjCurrent = NbProjection[comptUpdate-1]
        print("..... Initialize number of projections in DINCOnvAE model # %d"%(NbProjection[comptUpdate-1]))
        global_model_FP,global_model_FP_Masked = define_DINConvAE(NbProjection[comptUpdate-1],model_AE,\
                                                                  x_train.shape,\
                                                                  flag_MultiScaleAEModel,flagUseMaskinEncoder)
        if flagTrOuputWOMissingData == 1:
            global_model_FP.compile(loss='mean_squared_error',\
                                    optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate-1]))
        else:
            global_model_FP_Masked.compile(loss='mean_squared_error',\
                                    optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate-1]))

    # ******************** #
    # Start Learning model #
    # ******************** #
        
    print("..... Start learning AE model %d FP/Grad %d"%(flagAEType,flagOptimMethod))
    for iter in range(iterInit,Niter):
        if iter == IterUpdate[comptUpdate]:
            # update DINConvAE model
            NBProjCurrent = NbProjection[comptUpdate]
            print("..... Update/initialize number of projections in DINCOnvAE model # %d"%(NbProjection[comptUpdate]))
            global_model_FP,global_model_FP_Masked = define_DINConvAE(NbProjection[comptUpdate],model_AE,x_train.shape,\
                                                                          flag_MultiScaleAEModel,flagUseMaskinEncoder)
            if flagTrOuputWOMissingData == 1:
                global_model_FP.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
            else:
                global_model_FP_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))

            if comptUpdate < len(NbProjection)-1:
                comptUpdate += 1
        
        # gradient descent iteration            
        if flagTrOuputWOMissingData == 1:
            history = global_model_FP.fit([x_train_init,mask_train],x_train,
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

        x_train_pred    = global_model_FP.predict([x_train_init,mask_train])
        x_test_pred     = global_model_FP.predict([x_test_init,mask_test])

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
        rec_AE_Tr     = model_AE.predict([x_train,np.ones((mask_train.shape))])
        rec_AE_Tt     = model_AE.predict([x_test,np.ones((mask_test.shape))])
        
        exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
        
        print(".......... Auto-encoder performance when applied to gap-free data")
        print('.... explained variance AE (Tr)  : %.2f%%'%(100.*exp_var_AE_Tr))
        print('.... explained variance AE (Tt)  : %.2f%%'%(100.*exp_var_AE_Tt))
        
        if flagUseMaskinEncoder == 1:
            rec_AE_Tr     = model_AE.predict([x_train,np.zeros((mask_train.shape))])
            rec_AE_Tt     = model_AE.predict([x_test,np.zeros((mask_train.shape))])
        
            exp_var_AE_Tr,exp_var_AE_Tt = eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt)
        
            print('.... explained variance AE (Tr) with mask  : %.2f%%'%(100.*exp_var_AE_Tr))
            print('.... explained variance AE (Tt) with mask  : %.2f%%'%(100.*exp_var_AE_Tt))
        print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1]))
        print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))  

        # update training data
        if iter > IterUpdateInit:
            # mask = 0(missing data) ; 1(data)
            x_train_init = mask_train * x_train_missing + (1.-mask_train) * x_train_pred
            x_test_init  = mask_test  * x_test_missing  + (1.-mask_test)  * x_test_pred
            
        # save models
        genSuffixModel=save_Models(dict_global_Params,genFilename,NBProjCurrent,encoder,decoder,iter)

        # generate some plots
        plot_Figs(dirSAVE,genFilename,genSuffixModel,\
                  x_train,x_train_missing,x_train_pred,rec_AE_Tr,\
                  x_test,x_test_missing,lday_test,x_test_pred,rec_AE_Tt,iter)

        # Save DINAE result         
        saved_path = dirSAVE+'/saved_path_%03d'%(iter)+'.pickle'
        with open(saved_path, 'wb') as handle:
            pickle.dump([x_test,x_test_missing,x_test_pred,rec_AE_Tt], handle)

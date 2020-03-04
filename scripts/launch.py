#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:21:45 2019

@author: rfablet, mbeaucha
"""

from dinae import *

opt      = sys.argv[1]
lag      = sys.argv[2]
type_obs = sys.argv[3]
   
# main code
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # list of global parameters (comments to add)
    flagTrWMissingData          = 0     # Training phase with or without missing data
    flagloadOIData 		= 1     # load OI: work on rough variable or anomaly
    include_covariates          = True # use sst and sss
    size_tw                     = 11    # Length of the 4th dimension          
    Wsquare     		= 4     # half-width of holes
    Nsquare     		= 3     # number of holes
    DimAE       		= 100   # Dimension of the latent space
    flagAEType  		= 7     # model type, ConvAE or GE-NN
    flag_MultiScaleAEModel      = 0     # see flagProcess2_7: work on HR(0), LR(1), or HR+LR(2)
    flagOptimMethod 		= 0     # 0 DINAE : iterated projections, 1 : Gradient descent  
    flagGradModel   		= 0     # 0: F(Grad,Mask), 1: F==(Grad,Grad(t-1),Mask), 2: LSTM(Grad,Mask)
    sigNoise        		= 1e-1
    flagUseMaskinEncoder 	= 0
    flagTrOuputWOMissingData    = 1
    stdMask              	= 0.
    flagDataWindowing 		= 2  # 2 for SSH case-study
    dropout           		= 0.0
    wl2               		= 0.0000
    batch_size        		= 12
    NbEpoc            		= 20
    if flagTrWMissingData == 1:
        Niter = 20
    else:
        Niter = 20

    # create the output directory
    if flagAEType==6:
        suf1 = 'ConvAE'
    else:
        suf1 = 'GENN'
    if flagTrWMissingData==0:
        suf2 = 'WOmissing'
    else:
        suf2 = 'Wmissing'
    if flagOptimMethod==0:
        suf3 = 'FP'
    else:
        suf3 = 'GB'
    # output directory
    if opt!='swot':
        dirSAVE = '/mnt/groupadiag302/WG8/DINAE/resIA_'+opt+'_nadlag_'+lag+"_"+type_obs+"/"+suf3+'_'+suf1+'_'+suf2+'/'
    else:
        dirSAVE = '/mnt/groupadiag302/WG8/DINAE/resIA_'+opt+'_'+type_obs+"/"+suf3+'_'+suf1+'_'+suf2+'/'
    if not os.path.exists(dirSAVE):
        mk_dir_recursive(dirSAVE)
    else:
        shutil.rmtree(dirSAVE)
        mk_dir_recursive(dirSAVE)

    # push all global parameters in a list
    def createGlobParams(params):
        return dict(((k, eval(k)) for k in params))
    list_globParams=['flagTrOuputWOMissingData','flagTrWMissingData',\
    'flagloadOIData','include_covariates','size_tw','Wsquare',\
    'Nsquare','DimAE','flagAEType',\
    'flagOptimMethod','flagGradModel','sigNoise',\
    'flagUseMaskinEncoder','stdMask',\
    'flagDataWindowing','dropout','wl2','batch_size',\
    'NbEpoc','Niter',\
    'flag_MultiScaleAEModel','dirSAVE','suf1','suf2','suf3']
    globParams = createGlobParams(list_globParams)   

    #1) *** Read the data ***
    genFilename, x_train, y_train, mask_train, gt_train, x_train_missing, meanTr, stdTr,\
    x_test, y_test, mask_test, gt_test, x_test_missing, lday_test, x_train_OI, x_test_OI = flagProcess0(globParams,lag,opt,type_obs)

    #3) *** Define AE architecture ***
    genFilename, encoder, decoder, model_AE, DIMCAE = flagProcess2(globParams,genFilename,x_train,mask_train,x_test,mask_test)

    #4) *** Define classifier architecture for performance evaluation ***
    #classifier = flagProcess3(globParams,y_train)

    #5) *** Train ConvAE ***      
    if flagOptimMethod==0:
        flagProcess4_Optim0(globParams,genFilename,x_train,x_train_missing,mask_train,gt_train,meanTr,stdTr,\
                 x_test,x_test_missing,mask_test,gt_test,lday_test,x_train_OI,x_test_OI,encoder,decoder,model_AE,DIMCAE)
    if flagOptimMethod==1:
        flagProcess4_Optim1(globParams,genFilename,x_train,x_train_missing,mask_train,gt_train,meanTr,stdTr,\
                 x_test,x_test_missing,mask_test,gt_test,lday_test,x_train_OI,x_test_OI,encoder,decoder,model_AE,DIMCAE)


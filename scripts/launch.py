#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:21:45 2019

@author: rfablet, mbeaucha
"""

from dinae import *

def ifelse(cond1,val1,val2):
    if cond1==True:
        res = val1
    else:
        res = val2
    return res

opt      = sys.argv[1]
lag      = sys.argv[2]
type_obs = sys.argv[3]
   
# main code
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # list of global parameters (comments to add)
    fileMod                     = "/mnt/groupadiag302/WG8/maps/NATL60-CJM165_ssh_y2013.1y.nc" # Model file
    fileOI                      = "/mnt/groupadiag302/WG8/oi/ssh_NATL60_4nadir.nc"            # OI file
    if opt=="nadir":
        fileObs                 = "/mnt/groupadiag302/WG8/data/gridded_data_swot_wocorr/dataset_nadir_"+lag+"d.nc" # Obs file (1)
    elif opt=="swot": 
        fileObs                 = "/mnt/groupadiag302/WG8/data/gridded_data_swot_wocorr/dataset_swot.nc"           # Obs file (2)
    else:
        fileObs                 = "/mnt/groupadiag302/WG8/data/gridded_data_swot_wocorr/dataset_nadir_"+lag+"d_swot.nc" # Obs file (3)
    flagTrWMissingData          = 0     # Training phase with or without missing data
    flagloadOIData 		= 1     # load OI: work on rough variable or anomaly
    include_covariates          = False  # use additional covariates in initial layer
    N_cov                       = 3     # SST, SSS and OI
    lfile_cov                   = ["/mnt/groupadiag302/WG8/maps/NATL60-CJM165_sst_y2013.1y.nc",\
                                   "/mnt/groupadiag302/WG8/maps/NATL60-CJM165_sss_y2013.1y.nc",\
                                   "/mnt/groupadiag302/WG8/oi/ssh_NATL60_4nadir.nc"]
    lname_cov                   = ["sst","sss","ssh_mod"]
    lid_cov                     = ["SST","SSS","OI"]
    size_tw                     = 11    # Length of the 4th dimension          
    Wsquare     		= 4     # half-width of holes
    Nsquare     		= 3     # number of holes
    DimAE       		= 100   # Dimension of the latent space
    flagAEType  		= 6     # model type, ConvAE or GE-NN
    flagLoadModel               = 0     # load pre-defined AE model or not
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
    batch_size        		= 4
    NbEpoc            		= 20
    Niter = ifelse(flagTrWMissingData==1,40,40)

    # create the output directory
    suf1 = ifelse(flagAEType==6,"ConvAE","GENN")
    suf2 = ifelse(flagTrWMissingData==0,"WOmissing","Wmissing")
    suf3 = ifelse(flagOptimMethod==0,"FP","GB")
    dirSAVE = ifelse(opt!='swot',\
              '/mnt/groupadiag302/WG8/DINAE/resIA_'+opt+'_nadlag_'+lag+"_"+type_obs+"/"+suf3+'_'+suf1+'_'+suf2+'/',\
              '/mnt/groupadiag302/WG8/DINAE/resIA_'+opt+'_'+type_obs+"/"+suf3+'_'+suf1+'_'+suf2+'/')
    if not os.path.exists(dirSAVE):
        mk_dir_recursive(dirSAVE)
    else:
        shutil.rmtree(dirSAVE)
        mk_dir_recursive(dirSAVE)

    # push all global parameters in a list
    def createGlobParams(params):
        return dict(((k, eval(k)) for k in params))
    list_globParams=['fileMod','fileObs','fileOI',\
    'include_covariates','N_cov','lfile_cov','lid_cov','lname_cov',\
    'flagTrOuputWOMissingData','flagTrWMissingData',\
    'flagloadOIData','size_tw','Wsquare',\
    'Nsquare','DimAE','flagAEType','flagLoadModel',\
    'flagOptimMethod','flagGradModel','sigNoise',\
    'flagUseMaskinEncoder','stdMask',\
    'flagDataWindowing','dropout','wl2','batch_size',\
    'NbEpoc','Niter',\
    'flag_MultiScaleAEModel','dirSAVE','suf1','suf2','suf3']
    globParams = createGlobParams(list_globParams)   

    #1) *** Read the data ***
    genFilename, x_train, y_train, mask_train, gt_train, x_train_missing, meanTr, stdTr,\
    x_test, y_test, mask_test, gt_test, x_test_missing, lday_test, x_train_OI, x_test_OI = import_Data(globParams,type_obs)

    #3) *** Define AE architecture ***
    genFilename, encoder, decoder, model_AE, DIMCAE = define_Models(globParams,genFilename,x_train,mask_train,x_test,mask_test)

    #4) *** Define classifier architecture for performance evaluation ***
    #classifier = flagProcess3(globParams,y_train)

    #5) *** Train ConvAE ***      
    if flagOptimMethod == 0:
        flagProcess4_Optim0(globParams,genFilename,x_train,x_train_missing,mask_train,gt_train,meanTr,stdTr,\
                 x_test,x_test_missing,mask_test,gt_test,lday_test,x_train_OI,x_test_OI,encoder,decoder,model_AE,DIMCAE)
    if flagOptimMethod == 1:
        flagProcess4_Optim1(globParams,genFilename,x_train,x_train_missing,mask_train,gt_train,meanTr,stdTr,\
                 x_test,x_test_missing,mask_test,gt_test,lday_test,x_train_OI,x_test_OI,encoder,decoder,model_AE,DIMCAE)


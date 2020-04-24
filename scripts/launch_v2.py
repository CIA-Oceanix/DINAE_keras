#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:21:45 2019

@author: rfablet, mbeaucha
"""

from dinae_keras import *

def ifelse(cond1,val1,val2):
    if cond1==True:
        res = val1
    else:
        res = val2
    return res

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

opt      = "nadirswot"
lag      = "5"
type_obs = sys.argv[1]
domain   = sys.argv[4] 
  
# main code
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # list of global parameters (comments to add)
    fileMod                     = datapath+"/"+domain+"/maps/NATL60-CJM165_ssh_y2013.1y.nc" # Model file
    fileOI                      = datapath+"/"+domain+"/oi/ssh_NATL60_4nadir.nc"            # OI file
    if opt=="nadir":
        fileObs                 = datapath+"/"+domain+"/data/gridded_data_swot_wocorr/dataset_nadir_"+lag+"d.nc" # Obs file (1)
    elif opt=="swot": 
        fileObs                 = datapath+"/"+domain+"/data/gridded_data_swot_wocorr/dataset_swot.nc"           # Obs file (2)
    else:
        fileObs                 = datapath+"/"+domain+"/data/gridded_data_swot_wocorr/dataset_nadir_"+lag+"d_swot.nc" # Obs file (3)
    flagTrWMissingData          = int(sys.argv[2])     # Training phase with or without missing data
    if flagTrWMissingData==2:
        flagTrOuputWOMissingData = 0
    else :
        flagTrOuputWOMissingData = 1
    flagloadOIData 		= 1     # load OI: work on rough variable or anomaly
    include_covariates          = str2bool(sys.argv[3])  # use additional covariates in initial layer
    N_cov                       = 1     # SST, SSS and OI
    if include_covariates==False:
        N_cov = 0 
    '''N_cov                       = 3     # SST, SSS and OI
    lfile_cov                   = [datapath+"/"+domain+"/maps/NATL60-CJM165_sst_y2013.1y.nc",\
                                   datapath+"/"+domain+"/maps/NATL60-CJM165_sss_y2013.1y.nc",\
                                   datapath+"/"+domain+"/oi/ssh_NATL60_4nadir.nc"]
    lname_cov                   = ["sst","sss","ssh_mod"]
    lid_cov                     = ["SST","SSS","OI"]'''
    lfile_cov                   = [datapath+"/"+domain+"/oi/ssh_NATL60_4nadir.nc"]
    if type_obs=="mod":
        lname_cov               = ["ssh_mod"]
    else:
        lname_cov               = ["ssh_obs"]
    lid_cov                     = ["OI"]
    size_tw                     = 11    # Length of the 4th dimension          
    Wsquare     		= 4     # half-width of holes
    Nsquare     		= 3     # number of holes
    DimAE       		= 200   # Dimension of the latent space
    flagAEType  		= 7     # model type, ConvAE or GE-NN
    flagLoadModel               = 0     # load pre-defined AE model or not
    flag_MultiScaleAEModel      = 0     # see flagProcess2_7: work on HR(0), LR(1), or HR+LR(2)
    flagOptimMethod 		= 0     # 0 DINAE : iterated projections, 1 : Gradient descent  
    flagGradModel   		= 0     # 0: F(Grad,Mask), 1: F==(Grad,Grad(t-1),Mask), 2: LSTM(Grad,Mask)
    sigNoise        		= 1e-1
    flagUseMaskinEncoder 	= 0
    stdMask              	= 0.
    flagDataWindowing 		= 2  # 2 for SSH case-study
    dropout           		= 0.0
    wl2               		= 0.0000
    batch_size        		= 4
    NbEpoc            		= 20
    Niter = ifelse(flagTrWMissingData==1,20,20)

    # create the output directory
    suf1 = ifelse(flagAEType==6,"ConvAE","GENN")
    if flagTrWMissingData==0:
        suf2 = "womissing"
    elif flagTrWMissingData==1:
        suf2 = "wmissing"
    else:
        suf2 = "wwmissing"
    suf3 = ifelse(flagOptimMethod==0,"FP","GB")
    suf4 = ifelse(include_covariates==True,"w"+'-'.join(lid_cov),"wocov")
    dirSAVE = ifelse(opt!='swot',\
              '/gpfsscratch/rech/yrf/uba22to/DINAE/'+domain+'/resIA_'+opt+'_nadlag_'+lag+"_"+type_obs+"/"+suf3+'_'+suf1+'_'+suf2+'_'+suf4+'/',\
              '/gpfsscratch/rech/yrf/uba22to/DINAE/'+domain+'/resIA_'+opt+'_'+type_obs+"/"+suf3+'_'+suf1+'_'+suf2+'_'+suf4+'/')
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
    'NbEpoc','Niter','flag_MultiScaleAEModel',\
    'dirSAVE','suf1','suf2','suf3','suf4']
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


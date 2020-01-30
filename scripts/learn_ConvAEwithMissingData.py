#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:21:45 2019

@author: rfablet, mbeaucha
"""

from dinae import *
   
# main code
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # list of global parameters (comments to add)
    flagDisplay   		= 0
    flagSaveModel 		= 1
    flagTrOuputWOMissingData 	= 1
    flagloadOIData 		= 0
    flagDataset 		= 2
    Wsquare     		= 4#0 # half-width of holes
    Nsquare     		= 3  # number of holes
    DimAE       		= 40#20 # Dimension of the latent space
    flagAEType  		= 7#7#
    flagOptimMethod 		= 0 # 0 DINAE : iterated projections, 1 : Gradient descent  
    flagGradModel   		= 0 # 0: F(Grad,Mask), 1: F==(Grad,Grad(t-1),Mask), 2: LSTM(Grad,Mask)
    sigNoise        		= 1e-1
    flagUseMaskinEncoder 	= 0
    stdMask              	= 0.
    alpha                	= np.array([1.,0.,0.])
    flagDataWindowing 		= 2 # 2 for SST case-study
    dropout           		= 0.0
    wl2               		= 0.0000
    batch_size        		= 4
    NbEpoc            		= 20
    Niter             		= 50
    NSampleTr      		= 445
    flag_MultiScaleAEModel 	= 0 # see AE Type 007
    dirSAVE = '/mnt/groupadiag302/WG8/DINAE/'  
    # push all global parameters in a list
    def createGlobParams(params):
        return dict(((k, eval(k)) for k in params))
    list_globParams=['flagDisplay','flagSaveModel','flagTrOuputWOMissingData',\
    'flagloadOIData', 'flagDataset', 'Wsquare',\
    'Nsquare','DimAE','flagAEType',\
    'flagOptimMethod','flagGradModel','sigNoise',\
    'flagUseMaskinEncoder','stdMask','alpha',\
    'flagDataWindowing','dropout','wl2','batch_size',\
    'NbEpoc','Niter','NSampleTr',\
    'flag_MultiScaleAEModel','dirSAVE']
    globParams = createGlobParams(list_globParams)   

    #1) *** Read the data ***
    genFilename, x_train, y_train, mask_train, meanTr, stdTr, x_test, y_test, mask_test, lday_test = flagProcess0(globParams)
    #2) *** Generate missing data ***
    x_train_missing, x_test_missing = flagProcess1(globParams,x_train,mask_train,x_test,mask_test)
    #3) *** Define AE architecture ***
    genFilename, encoder, decoder, model_AE, DIMCAE = flagProcess2(globParams,genFilename,x_train,mask_train,x_test,mask_test)
    #4) *** Define classifier architecture for performance evaluation ***
    classifier = flagProcess3(globParams,y_train)
    #5) *** Train ConvAE ***      
    flagProcess4(globParams,genFilename,x_train,x_train_missing,mask_train,meanTr,stdTr,\
                 x_test,x_test_missing,mask_test,lday_test,encoder,decoder,model_AE,DIMCAE)

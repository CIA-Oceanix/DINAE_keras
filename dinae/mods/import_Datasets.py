from dinae import *

def Imputing_NaN(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell
    """
    if invalid is None: invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def flagProcess0(dict_global_Params,lag,opt,type_obs):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    #*** Start reading the data ***#
    thrMisData = 0.005
    # list of test dates
    indN_Tt = np.concatenate([np.arange(60,80),np.arange(140,160),\
                             np.arange(220,240),np.arange(300,320)])
    indN_Tr = np.delete(range(365),indN_Tt)
    #indN_Tr = np.arange(0,315)
    #indN_Tt = np.arange(315,365)
    lday_test=[ datetime.strftime(datetime.strptime("2012-10-01",'%Y-%m-%d')\
                          + timedelta(days=np.float64(i)),"%Y-%m-%d") for i in indN_Tt ]

    indLat     = np.arange(0,200)
    indLon     = np.arange(0,200)         
    fileMod = "/mnt/groupadiag302/WG8/maps/NATL60-CJM165_ssh_y2013.1y.nc"
    if opt=="nadir":
        fileObs = "/mnt/groupadiag302/WG8/data/gridded_data_swot_wocorr/dataset_nadir_"+lag+"d.nc"
    elif opt=="swot":
        fileObs = "/mnt/groupadiag302/WG8/data/gridded_data_swot_wocorr/dataset_swot.nc" 
    else:
        fileObs = "/mnt/groupadiag302/WG8/data/gridded_data_swot_wocorr/dataset_nadir_"+lag+"d_swot.nc"
    fileOI  = "/mnt/groupadiag302/WG8/oi/ssh_NATL60_4nadir.nc"
 
    #*** TRAINING DATASET ***#
    print("1) .... Load SST dataset (training data): "+fileObs)

    nc_data_mod = Dataset(fileMod,'r')
    nc_data_obs = Dataset(fileObs,'r')    
    x_orig      = np.copy(nc_data_mod['ssh'][:,indLon,indLat])
    for i in range(x_orig.shape[0]):
        x_orig[i,:,:] = Imputing_NaN(x_orig[i,:,:])
    # masking strategie differs according to flagTrWMissingData flag 
    mask_orig         = np.copy(nc_data_obs['ssh_mod'][:,indLon,indLat])
    mask_orig         = np.asarray(~np.isnan(mask_orig))
    if flagTrWMissingData==0:
        mask_orig[indN_Tr,:,:]  = 1
    if type_obs=="obs":
        noisy_obs = np.copy(nc_data_obs['ssh_obs'][:,indLon,indLat])
        obs       = np.copy(nc_data_obs['ssh_mod'][:,indLon,indLat])
        err_orig  = noisy_obs - obs
        if flagTrWMissingData==0:
            err_orig[indN_Tr,:,:] = 0.
        err_orig = np.where(np.isnan(err_orig), 0, err_orig)
    nc_data_mod.close()
    nc_data_obs.close()
    # load OI data
    if flagloadOIData == 1:
        print(".... Load OI SST dataset (training data): "+fileOI)
        nc_data    = Dataset(fileOI,'r')
        x_OI = np.copy(nc_data['ssh_mod'][:,indLon,indLat])
        for i in range(x_OI.shape[0]):
            x_OI[i,:,:] = Imputing_NaN(x_OI[i,:,:])
        nc_data.close()

    # create the time series (additional 4th time dimension)
    size_tw = 11
    x_train    = np.empty((len(indN_Tr),len(indLon),len(indLat),size_tw))
    x_train[:] = np.nan
    mask_train = np.empty((len(indN_Tr),len(indLon),len(indLat),size_tw))
    mask_train[:] = np.nan
    err_train = np.zeros((len(indN_Tr),len(indLon),len(indLat),size_tw))
    x_train_OI = np.empty((len(indN_Tr),len(indLon),len(indLat),size_tw))
    x_train_OI[:] = np.nan
    id_rm = []
    for k in range(len(indN_Tr)):
        idt = np.arange(indN_Tr[k]-np.floor(size_tw/2.),indN_Tr[k]+np.floor(size_tw/2.)+1,1)
        idt2= (np.where((idt>=0) & (idt<x_orig.shape[0]))[0]).astype(int)
        idt = (idt[idt2]).astype(int)
        if len(idt)<size_tw:
          id_rm.append(k)
        # build the training datasets
        if flagloadOIData == 1:
            x_train_OI[k,:,:,idt2] = x_OI[idt,:,:]
            x_train[k,:,:,idt2]    = x_orig[idt,:,:] - x_OI[idt,:,:]
        else:
            x_train[k,:,:,idt2]    = x_orig[idt,:,:]
        if type_obs=="obs":
            err_train[k,:,:,idt2] = err_orig[idt,:,:]
        mask_train[k,:,:,idt2] = mask_orig[idt,:,:]
    # Build ground truth data train
    gt_train = x_train
    # Build gappy (and potentially noisy) data train
    x_train_missing = (x_train * mask_train) + err_train
    if len(id_rm)>0:
        gt_train        = np.delete(gt_train,id_rm,axis=0)
        x_train         = np.delete(x_train,id_rm,axis=0)
        x_train_missing = np.delete(x_train_missing,id_rm,axis=0)
        mask_train      = np.delete(mask_train,id_rm,axis=0)
        x_train_OI      = np.delete(x_train_OI,id_rm,axis=0)
    print('.... # loaded samples: %d '%x_train.shape[0])

    # remove patch if no SSH data
    ss            = np.sum( np.sum( np.sum( x_train < -100 , axis = -1) , axis = -1 ) , axis = -1)
    ind           = np.where( ss == 0 )
    x_train         = x_train[ind[0],:,:,:]
    gt_train        = gt_train[ind[0],:,:,:]
    x_train_missing = x_train_missing[ind[0],:,:,:]
    mask_train      = mask_train[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_train_OI = x_train_OI[ind[0],:,:,:]
    rateMissDataTr_ = np.asarray(np.sum( np.sum( np.sum( mask_train , axis = -1) , axis = -1 ) , axis = -1), dtype=np.float64)
    rateMissDataTr_ /= mask_train.shape[1]*mask_train.shape[2]*mask_train.shape[3]
    ind        = np.where( rateMissDataTr_  >= thrMisData )
    gt_train        = gt_train[ind[0],:,:,:]
    x_train        = x_train[ind[0],:,:,:]
    x_train_missing = x_train_missing[ind[0],:,:,:]
    mask_train      = mask_train[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_train_OI = x_train_OI[ind[0],:,:,:]

    y_train = np.ones((x_train.shape[0]))

    if flagloadOIData:
        print("....... # of training patches: %d/%d"%(x_train.shape[0],x_train_OI.shape[0]))
    else:
        print("....... # of training patches: %d"%(x_train.shape[0]))
      
    # *** TEST DATASET ***#
    print("2) .... Load SST dataset (test data): "+fileObs)      

    # create the time series (additional 4th time dimension)
    x_test    = np.empty((len(indN_Tt),len(indLon),len(indLat),size_tw))
    x_test[:] = np.nan
    mask_test = np.empty((len(indN_Tt),len(indLon),len(indLat),size_tw))
    mask_test[:] = np.nan
    err_test = np.zeros((len(indN_Tt),len(indLon),len(indLat),size_tw))
    x_test_OI = np.empty((len(indN_Tt),len(indLon),len(indLat),size_tw))
    x_test_OI[:] = np.nan
    for k in range(len(indN_Tt)):
        idt = np.arange(indN_Tt[k]-np.floor(size_tw/2.),indN_Tt[k]+np.floor(size_tw/2.)+1,1)
        idt2= (np.where((idt>=0) & (idt<x_orig.shape[0]))[0]).astype(int)
        idt = (idt[idt2]).astype(int)
        if flagloadOIData == 1: 
            x_test_OI[k,:,:,idt2] = x_OI[idt,:,:]
            x_test[k,:,:,idt2]    = x_orig[idt,:,:] - x_OI[idt,:,:]
        else:
            x_test[k,:,:,idt2]    = x_orig[idt,:,:]
        if type_obs=="obs":
            err_test[k,:,:,idt2] = err_orig[idt,:,:]
        mask_test[k,:,:,idt2] = mask_orig[idt,:,:]
    # Build ground truth data test
    gt_test = x_test
    # Build gappy (and potentially noisy) data test
    x_test_missing = (x_test * mask_test) + err_test
    print('.... # loaded samples: %d '%x_test.shape[0])

    # remove patch if no SSH data
    ss            = np.sum( np.sum( np.sum( x_test < -100 , axis = -1) , axis = -1 ) , axis = -1)
    ind           = np.where( ss == 0 )
    x_test         = x_test[ind[0],:,:,:]
    gt_test        = gt_test[ind[0],:,:,:]
    x_test_missing = x_test_missing[ind[0],:,:,:]
    mask_test      = mask_test[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_test_OI = x_test_OI[ind[0],:,:,:]
    rateMissDataTr_ = np.asarray(np.sum( np.sum( np.sum( mask_test , axis = -1) , axis = -1 ) , axis = -1), dtype=np.float64)
    rateMissDataTr_ /= mask_test.shape[1]*mask_test.shape[2]*mask_test.shape[3]
    ind        = np.where( rateMissDataTr_  >= thrMisData )
    x_test         = x_test[ind[0],:,:,:]
    gt_test        = gt_test[ind[0],:,:,:]
    x_test_missing = x_test_missing[ind[0],:,:,:]
    mask_test      = mask_test[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_test_OI = x_test_OI[ind[0],:,:,:]

    y_test    = np.ones((x_test.shape[0]))

    if flagloadOIData:
        print("....... # of test patches: %d /%d"%(x_test.shape[0],x_test_OI.shape[0]))
    else:
        print("....... # of test patches: %d"%(x_test.shape[0]))

    print("... mean Tr = %f"%(np.mean(gt_train)))
    print("... mean Tt = %f"%(np.mean(gt_test)))
            
    print(".... Training set shape %dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
    print(".... Test set shape     %dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
    
    genFilename = 'modelNATL60_SSH_'+str('%03d'%x_train.shape[0])+str('_%03d'%x_train.shape[1])+str('_%03d'%x_train.shape[2])
        
    print('....... Generic model filename: '+genFilename)
      
    meanTr          = np.mean( x_train )
    stdTr           = np.sqrt( np.mean( x_train**2 ) )
    x_train         = (x_train - meanTr)/stdTr
    x_train_missing = (x_train_missing - meanTr)/stdTr
    gt_train        = (gt_train - meanTr)/stdTr
    x_test          = (x_test  - meanTr)/stdTr
    x_test_missing  = (x_test_missing - meanTr)/stdTr
    gt_test         = (gt_test - meanTr)/stdTr
    
    print('... Mean and std of training data: %f  -- %f'%(meanTr,stdTr))

    if flagDataWindowing == 1:
        HannWindow = np.reshape(np.hanning(x_train.shape[2]),(x_train.shape[1],1)) * np.reshape(np.hanning(x_train.shape[1]),(x_train.shape[2],1)).transpose() 

        x_train = np.moveaxis(np.moveaxis(x_train,3,1) * np.tile(HannWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
        gt_train = np.moveaxis(np.moveaxis(gt_train,3,1) * np.tile(HannWindow,(gt_train.shape[0],gt_train.shape[3],1,1)),1,3)
        x_train_missing = np.moveaxis(np.moveaxis(x_train_missing,3,1) * np.tile(HannWindow,(x_train_missing.shape[0],x_train_missing.shape[3],1,1)),1,3)

        x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(HannWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
        gt_test  = np.moveaxis(np.moveaxis(gt_test,3,1) * np.tile(HannWindow,(gt_test.shape[0],gt_test.shape[3],1,1)),1,3)
        x_test_missing = np.moveaxis(np.moveaxis(x_test_missing,3,1) * np.tile(HannWindow,(x_test_missing.shape[0],x_test_missing.shape[3],1,1)),1,3)

        print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
        print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

    elif flagDataWindowing == 2:
        EdgeWidth  = 4
        EdgeWindow = np.zeros((x_train.shape[1],x_train.shape[2]))
        EdgeWindow[EdgeWidth:x_train.shape[1]-EdgeWidth,EdgeWidth:x_train.shape[2]-EdgeWidth] = 1
        
        x_train = np.moveaxis(np.moveaxis(x_train,3,1) * np.tile(EdgeWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
        gt_train = np.moveaxis(np.moveaxis(gt_train,3,1) * np.tile(EdgeWindow,(gt_train.shape[0],gt_train.shape[3],1,1)),1,3)
        x_train_missing = np.moveaxis(np.moveaxis(x_train_missing,3,1) * np.tile(EdgeWindow,(x_train_missing.shape[0],x_train_missing.shape[3],1,1)),1,3)

        x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(EdgeWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
        gt_test  = np.moveaxis(np.moveaxis(gt_test,3,1) * np.tile(EdgeWindow,(gt_test.shape[0],gt_test.shape[3],1,1)),1,3)
        x_test_missing = np.moveaxis(np.moveaxis(x_test_missing,3,1) * np.tile(EdgeWindow,(x_test_missing.shape[0],x_test_missing.shape[3],1,1)),1,3)

        mask_train = np.moveaxis(np.moveaxis(mask_train,3,1) * np.tile(EdgeWindow,(mask_train.shape[0],x_train.shape[3],1,1)),1 ,3)
        mask_test  = np.moveaxis(np.moveaxis(mask_test,3,1) * np.tile(EdgeWindow,(mask_test.shape[0],x_test.shape[3],1,1)),1,3)
        print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
        print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
      
    print("... (after normalization) mean Tr = %f"%(np.mean(gt_train)))
    print("... (after normalization) mean Tt = %f"%(np.mean(gt_test)))
      
    return genFilename, x_train,y_train, mask_train, gt_train, x_train_missing, meanTr, stdTr, x_test, y_test, mask_test, gt_test, x_test_missing, lday_test, x_train_OI, x_test_OI


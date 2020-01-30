from dinae import *

def Imputing_NaN(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell
    """
    if invalid is None: invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def flagProcess0(dict_global_Params):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    #*** Start reading the data ***#
    thrMisData = 0.005
    # list of test dates
    indN_Tt = np.concatenate([np.arange(60,80),np.arange(140,160),\
                             np.arange(220,240),np.arange(300,320)])
    indN_Tr = np.delete(range(365),indN_Tt)
    lday_test=[ datetime.strftime(datetime.strptime("2012-10-01",'%Y-%m-%d')\
                          + timedelta(days=np.float64(i)),"%Y-%m-%d") for i in indN_Tt ]

    indLat     = np.arange(0,200)
    indLon     = np.arange(0,200)         
    fileMod = "/mnt/groupadiag302/WG8/maps/NATL60-CJM165_ssh_y2013.1y.nc"
    fileObs = "/mnt/groupadiag302/WG8/data/dataset_nadir_0d_swot.nc"
    fileOI  = "/mnt/groupadiag302/WG8/oi/ssh_NATL60_4nadir.nc"
 
    #*** TRAINING DATASET ***#
            
    print("1) .... Load SST dataset (training data): "+fileObs)
    nc_data_mod = Dataset(fileMod,'r')
    nc_data_obs = Dataset(fileObs,'r')    
    print('.... # samples: %d '%nc_data_mod.dimensions['time'].size)
    x_train        = nc_data_mod['ssh'][indN_Tr,indLon,indLat]
    for i in range(x_train.shape[0]):
        x_train[i,:,:] = Imputing_NaN(x_train[i,:,:])
    mask_train     = nc_data_obs['ssh_mod'][indN_Tr,indLon,indLat]
    mask_train     = np.asarray(~np.isnan(mask_train))
    print('.... # loaded samples: %d '%x_train.shape[0])
    nc_data_mod.close()
    nc_data_obs.close()
    
    x_train    = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
    mask_train = mask_train.reshape((mask_train.shape[0],mask_train.shape[1],mask_train.shape[2],1))

    # load OI data
    if flagloadOIData == 1:
        print(".... Load OI SST dataset (training data): "+fileOI)
        nc_data    = Dataset(fileOI,'r')
        x_train_OI = nc_data['ssh_mod'][indN_Tr,indLon,indLat]
        for i in range(x_train_OI.shape[0]):
            x_train_OI[i,:,:] = Imputing_NaN(x_train_OI[i,:,:])

    # remove patch if no SSH data
    ss            = np.sum( np.sum( np.sum( x_train < -100 , axis = -1) , axis = -1 ) , axis = -1)
    ind           = np.where( ss == 0 )
    x_train    = x_train[ind[0],:,:,:]
    mask_train = mask_train[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_train_OI = x_train_OI[ind[0],:,:]
    rateMissDataTr_ = np.asarray(np.sum( np.sum( np.sum( mask_train , axis = -1) , axis = -1 ) , axis = -1), dtype=np.float64)
    rateMissDataTr_ /= mask_train.shape[1]*mask_train.shape[2]*mask_train.shape[3]
    ind        = np.where( rateMissDataTr_  >= thrMisData )
    x_train_    = x_train[ind[0],:,:,:]
    mask_train_ = mask_train[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_train_OI = x_train_OI[ind[0],:,:]

    y_train = np.ones((x_train.shape[0]))

    if flagloadOIData:
        print("....... # of training patches: %d/%d"%(x_train.shape[0],x_train_OI.shape[0]))
    else:
        print("....... # of training patches: %d"%(x_train.shape[0]))
      
    # *** TEST DATASET ***#
    print("2) .... Load SST dataset (test data): "+fileObs)      

    nc_data_mod = Dataset(fileMod,'r')
    nc_data_obs = Dataset(fileObs,'r')
    print('.... # samples: %d '%nc_data_mod.dimensions['time'].size)
    x_test        = nc_data_mod['ssh'][indN_Tt,indLon,indLat]
    for i in range(x_test.shape[0]):
        x_test[i,:,:] = Imputing_NaN(x_test[i,:,:])
    mask_test     = nc_data_obs['ssh_mod'][indN_Tt,indLon,indLat]
    mask_test     = np.asarray(~np.isnan(mask_test))
    print('.... # loaded samples: %d '%x_test.shape[0])
    nc_data_mod.close()
    nc_data_obs.close()

    x_test    = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
    mask_test = mask_test.reshape((mask_test.shape[0],mask_test.shape[1],mask_test.shape[2],1))

    # load OI data
    if flagloadOIData == 1:
        print(".... Load OI SST dataset (test data): "+fileOI)
        nc_data   = Dataset(fileOI,'r')
        x_test_OI = nc_data['ssh_mod'][indN_Tt,indLon,indLat]         
        for i in range(x_test_OI.shape[0]):
            x_test_OI[i,:,:] = Imputing_NaN(x_test_OI[i,:,:])

    # remove patch if no SSH data
    ss            = np.sum( np.sum( np.sum( x_test < -100 , axis = -1) , axis = -1 ) , axis = -1)
    ind           = np.where( ss == 0 )
    x_test    = x_test[ind[0],:,:,:]
    mask_test = mask_test[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_test_OI = x_test_OI[ind[0],:,:]
    rateMissDataTr_ = np.asarray(np.sum( np.sum( np.sum( mask_test , axis = -1) , axis = -1 ) , axis = -1), dtype=np.float64)
    rateMissDataTr_ /= mask_test.shape[1]*mask_test.shape[2]*mask_test.shape[3]
    ind        = np.where( rateMissDataTr_  >= thrMisData )
    x_test_    = x_test[ind[0],:,:,:]
    mask_test_ = mask_test[ind[0],:,:,:]
    if flagloadOIData == 1:
        x_test_OI = x_test_OI[ind[0],:,:]

    y_test    = np.ones((x_test.shape[0]))

    if flagloadOIData:
        print("....... # of test patches: %d /%d"%(x_test.shape[0],x_test_OI.shape[0]))
    else:
        print("....... # of test patches: %d"%(x_test.shape[0]))

    print("... mean Tr = %f"%(np.mean(x_train)))
    print("... mean Tt = %f"%(np.mean(x_test)))
            
    print(".... Training set shape %dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
    print(".... Test set shape     %dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
    
    genFilename = 'modelNATL60_SSH_'+str('%03d'%x_train.shape[0])+str('_%03d'%x_train.shape[1])+str('_%03d'%x_train.shape[2])
        
    print('....... Generic model filename: '+genFilename)
      
    meanTr     = np.mean(x_train)
    x_train    = x_train - meanTr
    x_test     = x_test  - meanTr
    
    if flagloadOIData:
        x_train_OI    = x_train_OI - meanTr
        x_test_OI     = x_test_OI  - meanTr

    # scale wrt std
    stdTr      = np.sqrt( np.mean( x_train**2 ) )
    x_train    = x_train / stdTr
    x_test     = x_test  / stdTr

    print('... Mean and std of training data: %f  -- %f'%(meanTr,stdTr))

    if flagloadOIData == 1:
        x_train_OI    = x_train_OI / stdTr
        x_test_OI     = x_test_OI  / stdTr

    if flagDataWindowing == 1:
        HannWindow = np.reshape(np.hanning(x_train.shape[2]),(x_train.shape[1],1)) * np.reshape(np.hanning(x_train.shape[1]),(x_train.shape[2],1)).transpose() 

        x_train = np.moveaxis(np.moveaxis(x_train,3,1) * np.tile(HannWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
        x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(HannWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
        print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
        print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

    elif flagDataWindowing == 2:
        EdgeWidth  = 4
        EdgeWindow = np.zeros((x_train.shape[1],x_train.shape[2]))
        EdgeWindow[EdgeWidth:x_train.shape[1]-EdgeWidth,EdgeWidth:x_train.shape[2]-EdgeWidth] = 1
        
        x_train = np.moveaxis(np.moveaxis(x_train,3,1) * np.tile(EdgeWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
        x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(EdgeWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)

        mask_train = np.moveaxis(np.moveaxis(mask_train,3,1) * np.tile(EdgeWindow,(mask_train.shape[0],x_train.shape[3],1,1)),1,3)
        mask_test  = np.moveaxis(np.moveaxis(mask_test,3,1) * np.tile(EdgeWindow,(mask_test.shape[0],x_test.shape[3],1,1)),1,3)
        print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
        print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
      
    print("... (after normalization) mean Tr = %f"%(np.mean(x_train)))
    print("... (after normalization) mean Tt = %f"%(np.mean(x_test)))
      
    return genFilename, x_train,y_train, mask_train, meanTr, stdTr, x_test, y_test, mask_test, lday_test


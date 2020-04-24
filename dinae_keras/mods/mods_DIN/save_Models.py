from dinae_keras import *

def save_Models(dict_global_Params,genFilename,NBProjCurrent,encoder,decoder,iter,*args):   

    if len(args)>0:
        args[0] = gradModel
        args[1] = gradMaskModel
        args[2] = NBGradCurrent
    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    alpha=[1.,0.,0.]
    genSuffixModel = '_Alpha%03d'%(100*alpha[0]+10*alpha[1]+alpha[2])
    if flagUseMaskinEncoder == 1:
        genSuffixModel = genSuffixModel+'_MaskInEnc'
        if stdMask  > 0:
            genSuffixModel = genSuffixModel+'_Std%03d'%(100*stdMask)

    if flagOptimMethod == 0:
        if flagTrOuputWOMissingData == 1:
            genSuffixModel = genSuffixModel+'_AETRwoMissingData'+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))
        else:
            genSuffixModel = genSuffixModel+'_AE'+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))
    elif flagOptimMethod == 1:
        if flagTrOuputWOMissingData == 1:
            genSuffixModel = genSuffixModel+'GradAETRwoMissingData'+str('%02d'%(flagAEType))+str('_%02d'%(flagGradModel))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))+'_Grad'+str('%02d'%(NBGradCurrent))
        else:
            genSuffixModel = genSuffixModel+'GradAE'+str('%02d'%(flagAEType))+str('_%02d'%(flagGradModel))+'_D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))+'_Grad'+str('%02d'%(NBGradCurrent))

    fileMod = dirSAVE+genFilename+genSuffixModel+'_Encoder_iter%03d'%(iter)+'.mod'
    print('.................. Encoder '+fileMod)
    encoder.save(fileMod)
    fileMod = dirSAVE+genFilename+genSuffixModel+'_Decoder_iter%03d'%(iter)+'.mod'
    print('.................. Decoder '+fileMod)
    decoder.save(fileMod)
    if flagOptimMethod == 1:
        fileMod = dirSAVE+genFilename+genSuffixModel+'_GradModel_iter%03d'%(iter)+'.mod'
        print('.................. GradModel '+fileMod)
        gradModel.save(fileMod)
        fileMod = dirSAVE+genFilename+genSuffixModel+'_GradMaskModel_iter%03d'%(iter)+'.mod'
        print('.................. GradMaskModel '+fileMod)
        gradMaskModel.save(fileMod)

    return genSuffixModel

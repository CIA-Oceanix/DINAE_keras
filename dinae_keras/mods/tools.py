from dinae_keras import *

# function to create recursive paths
def mk_dir_recursive(dir_path):
    if os.path.isdir(dir_path):
        return
    h, t = os.path.split(dir_path)  # head/tail
    if not os.path.isdir(h):
        mk_dir_recursive(h)

    new_path = join_paths(h, t)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)

def Gradient(img, order):
    """ calcuate x, y gradient and magnitude """ 
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobelx = sobelx/8.0
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobely = sobely/8.0
    sobel_norm = np.sqrt(sobelx*sobelx+sobely*sobely)
    if (order==0):
        return sobelx
    elif (order==1):
        return sobely
    else:
        return sobel_norm

def keras_custom_loss_function(size_tw):
    def insert_Sobel(size_tw,dir="x"):
        kernel_weights=np.zeros((3,3,size_tw,size_tw))
        if dir=="x":
            sobel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).T
        if dir=="y":
            sobel=np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).T
        for i in range(size_tw):
            kernel_weights[:,:,i,i]=sobel
        return kernel_weights
    def lossFunction(y_true,y_pred):
        mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        filter_gx  = K.constant(insert_Sobel(size_tw,"x"))
        filter_gy  = K.constant(insert_Sobel(size_tw,"y"))
        Gx_true = K.conv2d(y_true, filter_gx, padding="same")
        Gy_true = K.conv2d(y_true, filter_gy, padding="same")
        Grad_true = K.sqrt(keras.layers.Add()([keras.layers.Multiply()([Gx_true,Gx_true]),\
                                       keras.layers.Multiply()([Gy_true,Gy_true])]))
        Gx_pred = K.conv2d(y_pred, filter_gx, padding="same")
        Gy_pred = K.conv2d(y_pred, filter_gy, padding="same")
        Grad_pred = K.sqrt(keras.layers.Add()([keras.layers.Multiply()([Gx_pred,Gx_pred]),\
                                       keras.layers.Multiply()([Gy_pred,Gy_pred])]))
        mae_grad = tf.keras.losses.mean_absolute_error(Grad_true, Grad_pred)
        alpha= 0.5
        loss = ((1.-alpha)*mae) + (alpha*mae_grad)
        return loss
    return lossFunction

def thresholding(x,thr):
    greater = K.greater_equal(x,thr) #will return boolean values
    greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1    
    return greater



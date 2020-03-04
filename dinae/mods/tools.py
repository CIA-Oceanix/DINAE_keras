from dinae import *

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
    def lossFunction(y_true,y_pred):
        mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        filter_gx  = K.constant(np.dstack([-1,0,1,-2,0,2,-1,0,1]*size_tw),shape = [3, 3, size_tw, 1])
        filter_gy  = K.constant(np.dstack([-1,-2,-1,0,0,0,1,2,1]*size_tw),shape = [3, 3, size_tw, 1])
        Gx_true = K.conv2d(y_true, filter_gx)
        Gx_pred = K.conv2d(y_pred, filter_gx)
        Gy_true = K.conv2d(y_true, filter_gy)
        Gy_pred = K.conv2d(y_pred, filter_gy)
        Grad_true = K.sqrt(keras.layers.Add()([keras.layers.Multiply()([Gx_true,Gx_true]),\
                                           keras.layers.Multiply()([Gy_true,Gy_true])]))
        Grad_pred = K.sqrt(keras.layers.Add()([keras.layers.Multiply()([Gx_pred,Gx_pred]),\
                                           keras.layers.Multiply()([Gy_pred,Gy_pred])]))
        mae_grad = tf.keras.losses.mean_absolute_error(Grad_true, Grad_pred)
        alpha= 0.5
        loss = ((1.-alpha)*mae) + (alpha*mae_grad)
        return loss
    return lossFunction

def keras_custom_loss_function_v2(size_tw):
    # Create a gradient function
    def gradient(x,size_tw):
        # variance of the gradient  
        filter_gx = np.array([[[0.,0.,0.],[-0.5,0.,0.5],[0.,0.,0.]]]).reshape((3,3,1,1))
        filter_gx = np.dstack([filter_gx]*size_tw)
        gx        = keras.layers.Conv2D(1,(3,3),weights=[filter_gx],padding='same',activation='linear',use_bias=False,name='gx')(x)
        filter_gy = np.array([[[0.,-0.5,0.],[0.,0.,0.],[0.,0.5,0.]]]).reshape((3,3,1,1))
        filter_gy = np.dstack([filter_gy]*size_tw)
        gy        = keras.layers.Conv2D(1,(3,3),weights=[filter_gy],padding='same',activation='linear',use_bias=False,name='gy')(x)
        grad      = keras.layers.Add()([keras.layers.Multiply()([gx,gx]),keras.layers.Multiply()([gy,gy])])
        grad      = K.sqrt(grad)
    def lossFunction(y_true,y_pred):
        # Create a loss function that adds the MSE loss on the training variable and its gradient
        loss = K.mean(K.square(y_pred - y_true), axis=-1) + K.mean(K.square(gradient(y_pred,size_tw) - gradient(y_true,size_tw)) , axis=-1)
        # Return the loss function value
        return loss
    return lossFunction

def thresholding(x,thr):
    greater = K.greater_equal(x,thr) #will return boolean values
    greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1    
    return greater



import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
  
    kernelCenterY = (Hk-1) // 2
    kernelCenterX = (Wk-1) // 2
    for i in range(0,Hi):
        for j in range(0,Wi):
            for m in range(0,Hk):
                mm = Hk - 1 - m #index of kernel row (kernel should be flipped)
                for n in range(0,Wk):
                    nn = Wk - 1 - n #index of kernel col (kernel should be flipped)
                    ii = i + (m - kernelCenterX) 
                    jj = j + (n - kernelCenterY)
                    if ii >= 0 and ii < Hi and jj >= 0 and jj < Wi:
                        out[i][j] += image[ii][jj] * kernel[mm][nn]
  
    '''
    kernel = np.flip(np.flip(kernel,axis=0),axis = 1)
    delta_H = Hk // 2
    delta_W = Wk // 2
    for image_row in range(0 , Hi ):
        for image_col in range(0 , Wi ):
            for kernel_row in range(-delta_H,delta_H + 1):
                for kernel_col in range(-delta_W,delta_W + 1):
                    ii = image_row + kernel_row
                    jj = image_col + kernel_col
                    if ii >= 0 and ii < Hi and jj >= 0 and jj < Wi:
                        out[image_row][image_col] +=  image[ii][jj] * kernel[delta_H + kernel_row][delta_W + kernel_col]
    '''
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height:H + pad_height,pad_width:W + pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    deltaH =  (Hk-1) // 2 
    deltaW =  (Wk-1) // 2 
    pad_img = zero_pad(image,deltaH,deltaW)
    kernel = np.flip(np.flip(kernel,axis = 0),axis = 1)
    padH , padW = pad_img.shape
    for i in range(deltaH,padH - deltaH):
        for j in range(deltaW,padW - deltaW):
            out[i - deltaH][j - deltaW] = np.sum(pad_img[i-deltaH:i+ deltaH +1 , j-deltaW :j +  deltaW +1 ] * kernel)
    ### END YOUR CODE
    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
  
    fftsize = [Hi+Hk-1,Wi+Wk-1]
    fftImg = np.fft.fft2(image,fftsize )
    fftKer = np.fft.fft2(kernel,fftsize )
    convImg = np.fft.ifft2(fftImg*fftKer ).real
    out = convImg[Hk//2:convImg.shape[0]-Hk//2,Wk//2:convImg.shape[1]-Wk//2]
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
   
    ### YOUR CODE HERE
    g = np.flip(np.flip(g,axis=0),axis=1)
    out = conv_faster(f,g )
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    
    out = None
    ### YOUR CODE HERE
    g = g - np.mean(g)
    g = np.flip(np.flip(g,axis=0),axis=1)
    out = conv_faster(f,g )
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    # We don't support even kernel dimensions
    if g.shape[0] % 2 == 0:
        g = g[0:-1]
    if g.shape[1] % 2 == 0:
        g = g[:,0:-1]
    assert g.shape[0] % 2 == 1 and g.shape[1] % 2 == 1, "Even dimensions for filters is not allowed!"
    Hf,Wf = f.shape
    Hg,Wg = g.shape
    out = np.zeros((Hf,Wf))
    normalize_filter = (g - np.mean(g)) / np.std(g)
    deltaH =  (Hg-1) // 2 
    deltaW =  (Wg-1) // 2 
    pad_img = zero_pad(f,deltaH,deltaW)
    padH , padW = pad_img.shape

    for i in range(deltaH,padH - deltaH):
        for j in range(deltaW,padW - deltaW):
            src = pad_img[i-deltaH:i+ deltaH +1 , j-deltaW :j +  deltaW +1 ]
            normalize_src = (src - np.mean(src)) / np.std(src)
            out[i - deltaH][j - deltaW] = np.sum(normalize_src * normalize_filter)
    ### END YOUR CODE

    return out

import numpy as np
import cv2

def get_cosine_window(sz, method='hanning'):
    
    
    """
        "Finally, the image is multiplied by a cosine window which gradually reduces 
        the pixel values near the edge to zero. This also has the benefit that it 
        puts more emphasis near the center of the target."
        - Visual Object Tracking using Adaptive Correlation Filters
    """
    
    w, h = sz

    if method == 'blackman':
        w_win = np.blackman(w)
        h_win = np.blackman(h)

    elif method == 'hanning':
        w_win = np.hanning(w)
        h_win = np.hanning(h)

    elif method == 'hammming':
        w_win = np.hamming(w)
        h_win = np.hamming(h)

    w_msk, h_msk = np.meshgrid(w_win, h_win)
    win = w_msk * h_msk

    return win



def get_affine_transform(frame):
    
    a = -180 / 18
    b = 180 / 18
    r = a + (b - a) * np.random.uniform()
    
    h, w = frame.shape[1], frame.shape[0]

    rot_window = cv2.getRotationMatrix2D((h//2, w//2), r, 1)
    
    rot_frame = cv2.warpAffine(np.uint8(frame), rot_window, (h, w))
    rot_frame = rot_frame.astype(np.float32)
    
    return rot_frame



def clip_bbox(x, y, w, h, sz):
    
    x1 = int(np.clip(x, 0, sz[1]))
    x2 = int(np.clip(x+w, 0, sz[1]))
    y1 = int(np.clip(y, 0, sz[0]))
    y2 = int(np.clip(y+h, 0, sz[0]))
    
    return x1, y1, x2, y2




def log_transform(frame, eps = 1):
    """
        "First, the pixel values are transformed using a log function 
        which helps with low contrast lighting situations." 
        - Visual Object Tracking using Adaptive Correlation Filters
        
        Adding a smoothing parameter 1 as log is not defined in 0
        
    """
    return np.log(frame + eps)
    
    
def normalize_pixels(frame, eps = 1e-3):
        
    """
        "The pixel values are normalized to have a mean value of 0.0 and a norm of 1.0" 
        - Visual Object Tracking using Adaptive Correlation Filters
    """
        
    return (frame - np.mean(frame)) / (np.std(frame) + eps)



def scale_pixels(frame):
    return (frame - frame.min()) / (frame.max() - frame.min())


def scale_pixels_cw(frame):
    num = (frame - np.min(np.min(frame, axis = 0), axis=0))
    denom = (np.max(np.max(frame, axis = 0), axis=0)- np.min(np.min(frame, axis = 0), axis=0))
    return num/denom


def area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)

    if (dx >= 0) and (dy >= 0):
        return dx * dy

    return 0


def fft2(x):
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)

def ifft2(x):
    return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)
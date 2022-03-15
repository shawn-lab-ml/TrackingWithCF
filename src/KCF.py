import numpy as np
import cv2
from src.utils import *
from cyvlfeat.hog import hog
import math

class KCF():
    def __init__(self, padding=2.5, features='color', kernel='linear'):
        
        """
        padding : The tracked region has 2.5 times the size of the target 
        to provide some context and additional negative samples
        
        features: choice of features
        kernel: choice of kernel
        """
        
        self.padding = padding 
        self.lambda_r = 1e-4  #regularization
        self.features = features 
        self.kernel = kernel
        self.output_sigma_factor = 0.01
        
        if self.features=='hog':
            self.learning_rate = 0.08
            self.feature_bandwidth = 0.5
            self.cell_size=4
            
        elif self.features=='gray' or self.features=='color':
            self.learning_rate=0.15
            self.feature_bandwidth = 0.2
            self.cell_size=1

    def initialization(self,frame1,bbox):
        
        self.w, self.h = bbox[2], bbox[3]
        self.x0, self.y0 = bbox[0], bbox[1]
        
        
        # cell size of 4 pixels for HOG features
        self.padded_w = math.floor(self.w*(1+self.padding))//self.cell_size
        self.padded_h = math.floor(self.h *(1+self.padding))//self.cell_size
        
        
        self.x1_clip, self.y1_clip, self.x2_clip, self.y2_clip = clip_bbox(self.x0 - self.w*self.padding//2, 
                                                                           self.y0 - self.h*self.padding//2, 
                                                                           math.floor(self.w*(1+self.padding)), 
                                                                           math.floor(self.h*(1+self.padding)), 
                                                                           frame1.shape[:2])
        
        f = frame1[self.y1_clip:self.y2_clip, 
               self.x1_clip:self.x2_clip,
               :]
        
        # cosine window, which smoothly removes discontinuities at the image boundaries 
        # caused by the cyclic assumption
        self.cosine_window = get_cosine_window((self.padded_w, self.padded_h))
        self.cosine_window = np.expand_dims(self.cosine_window, axis=2)
        
        #self.spatial_bandwidth = self.output_sigma_factor * np.sqrt(self.padded_w * self.padded_h)/(self.cell_size*(1+self.padding))
        
        self.spatial_bandwidth = self.output_sigma_factor * np.sqrt(self.padded_w * self.padded_h)
        
        
        # Recall that the training samples consist of shifts of a base sample, so we must specify a regression target for each one in y. 
        # The regression targets y simply follow a Gaussian function, which takes a value of 1 for a centered target, 
        # and smoothly decays to 0 for any other shifts, according to the spatial bandwidth s
        self.yf = fft2(self._get_cyclic_gaussian_map())
        
        # Feature extraction
        self.xf = self._preprocess_frame(f)
        self.alphaf = self._train(self.xf,self.yf)
        
        

    def track(self,frame):
        
        f = frame[self.y1_clip:self.y2_clip, 
               self.x1_clip:self.x2_clip,
               :]
        
            
        z = self._preprocess_frame(f)
        

        responses = self._detect(self.alphaf, self.xf, z)
        
        max_yx = np.where(responses == np.max(responses))
        max_yx = (max_yx[0][0], max_yx[1][0])
        
        if max_yx[0]+1>self.padded_h/2:
            dy=max_yx[0]-self.padded_h
        else:
            dy=max_yx[0]
        if max_yx[1]+1> self.padded_w/2:
            dx=max_yx[1]-self.padded_w
        else:
            dx=max_yx[1]
        
        dy,dx = dy*self.cell_size, dx*self.cell_size
        self.x0 += dx
        self.y0 += dy
        
        
        self.x1_clip, self.y1_clip, self.x2_clip, self.y2_clip = clip_bbox(self.x0 - self.w*self.padding//2, 
                                                                           self.y0 - self.h*self.padding//2, 
                                                                           math.floor(self.w*(1+self.padding)), 
                                                                           math.floor(self.h*(1+self.padding)), 
                                                                           frame.shape[:2])
        
        f = frame[self.y1_clip:self.y2_clip, 
               self.x1_clip:self.x2_clip,
               :]
        
        
        new_x = self._preprocess_frame(f)
        new_alphaf = self._train(new_x, self.yf)
        
        self.alphaf = self.learning_rate * new_alphaf + (1 - self.learning_rate) * self.alphaf
        self.xf = self.learning_rate * new_x + (1 - self.learning_rate) * self.xf
        
        return self.x0, self.y0, self.x0 + self.w, self.y0 + self.h

    def _kernel_correlation(self, x1f, x2f):
        if self.kernel== 'gaussian':
            
            N=x1f.shape[0]*x1f.shape[1]
            
            xx=(np.dot(x1f.flatten().conj().T,x1f.flatten())/N)
            yy=(np.dot(x2f.flatten().conj().T,x2f.flatten())/N)
            cf=x1f*np.conj(x2f)
            c=np.sum(np.real(ifft2(cf)),axis=2)
            
            kf = fft2(np.exp(-1 / self.feature_bandwidth ** 2 * np.abs(xx+yy-2*c) / np.size(x1f)))
            
        elif self.kernel== 'linear':
            kf= np.sum(x1f*np.conj(x2f),axis=2)/np.size(x1f)

        return kf
    
    
    def _extract_features(self, x):
        
        if self.features=='gray' or self.features=='color':
            x = x / 255
            x = x-np.mean(x)
            
        elif self.features=='hog':

            x = hog(x/255, self.cell_size, variant = 'UoCTTI', bilinear_interpolation = True, n_orientations = 11)
            # clipping the values
           
            
        return x

    def _train(self, xf, yf):
        kf = self._kernel_correlation(xf, xf)
        alphaf = yf/(kf+self.lambda_r)
        return alphaf

    def _detect(self, alphaf, xf, zf):
        kf = self._kernel_correlation(zf, xf)
        responses = np.real(ifft2(alphaf * kf))
        return responses
    
    def _preprocess_frame(self, frame):
        
        x = cv2.resize(frame,(self.padded_w*self.cell_size,self.padded_h*self.cell_size))
        x = self._extract_features(x)

        return fft2(x*self.cosine_window)
    
    def _get_cyclic_gaussian_map(self):
        
        """since we always deal with cyclic signals,
        the peak of the Gaussian function must wrap around 
        from the top-left corner to the other corners"""
        
        
        # getting the distance from the center from the center
        # high values in the center of the gaussian map vs low values 
        # the further away you are
        xx, yy = np.meshgrid(np.arange(self.padded_w)-self.padded_w//2, 
                             np.arange(self.padded_h)-self.padded_h//2
                            )
                          
        dist = (xx**2+yy**2) / (self.spatial_bandwidth**2)
        
        # response_map
        response = np.exp(-0.5*dist)
        
        # allows us to have a response map with the aximum value being on the left corner
        # which represents the center of the ROI
        response = np.roll(response, -math.floor(self.padded_w/ 2), axis=1)
        response = np.roll(response,-math.floor(self.padded_h/2), axis=0)
        
        
        assert(response[0,0] == 1)
        
        return response
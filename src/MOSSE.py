import numpy as np
import cv2
from src.utils import *

class MOSSE:
    
    def __init__(self, lr = 1e-2, gauss_sig = 2, n_train = 100):     
        
        """
        lr : is the learning rate for the updating of A and B
        gauss_sig: is the standard deviation used for the response map in the distance calculation
        n_train: number of times to train on the first image using random affine transformations
        """
        
        self.lr = lr
        self.gauss_sig = gauss_sig
        self.n_train = n_train
        
        
    def initialization(self, frame1, bbox):
        
        """
        To be called on the first frame where the bbox is available.
        The tracker is trained on the first frame n_train times to optimal
        values for both A and B.
        """
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame1 = frame1.astype(np.float32)
        
        self.w, self.h = bbox[2], bbox[3]
        self.x0, self.y0 = bbox[0], bbox[1]
        
        self.x1_clip, self.y1_clip, self.x2_clip, self.y2_clip = clip_bbox(self.x0, 
                                                                           self.y0, 
                                                                           self.w, 
                                                                           self.h, 
                                                                           frame1.shape)
        
        self.cosine_window = get_cosine_window((self.w, self.h))
        
        g = self._get_gaussian_map()
        G = fft2(g)
        
        f = frame1[self.y1_clip:self.y2_clip, 
                   self.x1_clip:self.x2_clip]
        F = self._preprocess_frame(f) 
        
        self.A = G * np.conjugate(F)
        self.B = F * np.conjugate(F)
        
        self._train(f, G)
        
        self.A = self.lr * self.A
        self.B = self.lr * self.B
        
        
    def track(self, frame):
        
        """
        Based on the previous frame's bbox position, we predict the position
        of the next bbox.
        """
        
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32)    
        H = self.A/self.B
            
        f = frame[self.y1_clip:self.y2_clip, self.x1_clip:self.x2_clip]
        F = self._preprocess_frame(f)
            
        G = H * F
        
        g = scale_pixels(ifft2(G))
        max_yx = np.where(g == np.max(g))
        
        # updating the position of the bbox
        dy = int(np.mean(max_yx[0]) - g.shape[0] / 2)
        dx = int(np.mean(max_yx[1]) - g.shape[1] / 2)
            
        self.x0 += dx
        self.y0 += dy
        
        
        # make sure the values are within the frame
        self.x1_clip, self.y1_clip, self.x2_clip, self.y2_clip = clip_bbox(self.x0, 
                                                                           self.y0, 
                                                                           self.w, 
                                                                           self.h, 
                                                                           frame.shape)
        # Get the current frame with new bbox
        f = frame[self.y1_clip:self.y2_clip, 
                  self.x1_clip:self.x2_clip]
        F = self._preprocess_frame(f)
        
        # update A and B
        self.A = self.lr * (G * np.conjugate(F)) + (1 - self.lr) * self.A
        self.B = self.lr * (F * np.conjugate(F)) + (1 - self.lr) * self.B
        
        
        return self.x0, self.y0, self.x0+self.w, self.y0+self.h
            
    
    def _train(self, frame, G):
        
        
        """
        The objective is to initialize A and B through training on various images 
        which went through dome affine transformation.
        
        The filters were initialized by applying random small affine perturbations 
        to the tracking window for the first frame of the video
         - Visual Object Tracking using Adaptive Correlation Filters
        """
        
        for i in range(self.n_train):
            
            F = self._preprocess_frame(get_affine_transform(frame))
            self.A = self.A + G * np.conjugate(F)
            self.B = self.B + F * np.conjugate(F)
    
    def _preprocess_frame(self, frame):
        
        frame = cv2.resize(frame, (self.w, self.h))
        frame = log_transform(frame)
        frame = normalize_pixels(frame)
        
        return fft2(self.cosine_window * frame)

    
    def _get_gaussian_map(self):
        """
        "In this case, gi is generated from ground truth such that 
        it has a compact (σ = 2.0) 2D Gaussian shaped peak centered 
        on the target in training image fi."
        - Visual Object Tracking using Adaptive Correlation Filters
        """
        
        """h,w = frame.shape
        xx, yy = np.meshgrid(np.arange(self.w), np.arange(self.h))"""
        
        # getting the distance from the center from the center
        # high values in the center of the gaussian map vs low values 
        # the further away you are
        
        xx,yy = np.meshgrid(np.arange(self.w)-self.w//2, np.arange(self.h)-self.h//2)
        dist = (xx**2+yy**2) / (2*self.gauss_sig**2)
      
        # response_map
        
        response = np.exp(-0.5*dist)
        response = scale_pixels(response)
        
        return response
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:18:55 2023

@author: vikas
"""

import sys 
sys.path.insert(0, '../Utils' )
import tensorflow as tf

from NeuralNet_optimizer import Seq_NN, PhysicsInformedNN


#%%

class neural_net(Seq_NN):
    
    def __init_(self, ub, lb, layers):
        super(neural_net, self).__init__(ub, lb, layers)
    
    def call(self, inputs, training=False):
        output = super(neural_net, self).call(inputs)
        
        return output[:,0], output[:,1]
    
    
class Data_NN(PhysicsInformedNN):
    
    def __init__(self, 
                 x_f,
                 X_data, 
                 ub, lb, 
                 layers,
                 scale,
                 scale_out,
                 theta_data,
                 U_data,
                 rho_data,
                 alpha,
                 RePr):
        
        super(Data_NN, self).__init__()
        
        # Network architectures
        
        self.model = neural_net(ub, lb, layers)
        
        # Data initialization
        
        self.scale_max = scale[0]
        self.scale_min = scale[1]
        self.scale_out = scale_out
        
        self.theta_data = theta_data
        self.U_data = U_data
        self.rho_data = rho_data
        
        self.x_f = x_f
        self.X_data = X_data
        
        self.alpha = alpha
        
        self.RePr = RePr
        
        
    def loss(self):
        
        alpha = self.alpha
        theta_data = self.theta_data
        # rho_data = 1/(1 + alpha*theta_data/(1 - alpha))
        # u_data = 1.1435*0.264/rho_data
        rho_data = self.rho_data
        u_data = self.U_data
    
        
        theta_data_pred, u_data_pred = self.net_data_loss(self.x_f)
        
        theta_data_scaled = theta_data*self.scale_out[0] - self.scale_out[1]
        u_data_scaled = ((u_data - self.scale_min[0])/(self.scale_max[0] - 
                                                      self.scale_min[0]))*self.scale_out[0] - self.scale_out[1]
        # rho_data_scaled = ((u_data - self.scale_min[1])/(self.scale_max[1] - 
        #                                               self.scale_min[1]))*self.scale_out[0] - self.scale_out[1]
        
        yD = tf.reduce_mean(tf.square(theta_data_pred - theta_data_scaled)) + \
            0*tf.reduce_mean(tf.square(u_data_pred - u_data_scaled)) # +\
                # tf.reduce_mean(tf.square(rho_data_pred - rho_data_scaled)) 
                
        total_loss = yD
        emp = tf.constant(0)
            
        return total_loss, emp, emp, emp , emp, emp, emp, emp, emp, emp

    def net_data_loss(self, x):
        
        # tf.print(tf.reduce_min(y))
        
        # X = tf.stack([x, y], axis=1) # shape = (N_f,2)
        
        theta1, u1 = self.model(x)
        theta = (theta1 + self.scale_out[1])/self.scale_out[0]
        u = (u1 + self.scale_out[1])*(self.scale_max[0] - self.scale_min[0])/self.scale_out[0] + self.scale_min[0]

        return theta1, u1

   

    # For the final prediction 
    def predict(self, x):
        
        
        # X = tf.stack([x, y], axis=1) # shape = (N_f,2)
            
        theta1, u1 = self.model(x)     
        
        theta = (theta1 + self.scale_out[1])/self.scale_out[0]
        u = (u1 + self.scale_out[1])*(self.scale_max[0] - self.scale_min[0])/self.scale_out[0] + self.scale_min[0]
    
    
        return theta, u
 
    

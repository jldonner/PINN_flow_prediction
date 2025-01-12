#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:41:17 2022

@author: vikas
"""

import sys 
sys.path.insert(0, '../../optimizer' )
import tensorflow as tf
import numpy as np
from NeuralNet_optimizer import Seq_NN, PhysicsInformedNN


#%%

class neural_net(Seq_NN):
    
    def __init_(self, ub, lb, layers, hid_activation, out_activation):
        super(neural_net, self).__init__(ub, lb, layers, hid_activation, out_activation)
    
    def call(self, inputs, training=False):
        output = super(neural_net, self).call(inputs)
        
        return output[:,0]
    
    
class PINN_PDE(PhysicsInformedNN):
    '''
    The current class incorporates the boundary, and PDE losses.
    The boundary loss includes initial condition loss as well but one can also 
    create separate data set for and save the loss separately 
    You can always make changes to the class to incorporate the data loss, etc.
    
    
    In the current example, we are solving the 1D Burgers problem:
        Only 1 equation is needed for this problem        
        We use the dimensional form of the equations (some times non-dimensional 
                                                      form are also used in PINNs)
    
    At the end, the total loss is evaluated as the linear combination of the losses
    '''
    
    
    def __init__(self, 
                 hid_activation,
                 out_activation, 
                 x_t_boundary,
                 u_boundary,
                 x_t_colloc,
                 ub, lb, 
                 layers,
                 nu
                 ):
        
        super(PINN_PDE, self).__init__()
        
        # Network architectures
        
        self.model = neural_net(ub, lb, layers,  hid_activation, out_activation)
        
        # Data initialization        
        self.x_t_boundary = x_t_boundary
        self.u_boundary = u_boundary
        self.x_t_colloc = x_t_colloc
        self.ub = ub[0]
        self.lb = lb[0]
        self.layers = layers
        self.nu = nu
        

        
        
    def loss(self):
        

        u_boundary = self.u_boundary  
                
        # We use only one equations here 
        pde_loss = self.net_equation_loss()
                
        
        # PDE loss mse
        yS = 1*tf.reduce_mean(tf.square(pde_loss))
        
        
        # Loss from boundary and initial conditions (provided in the model as training data)
        u_boundary_pred = self.net_boundary_loss(self.x_t_boundary)
        
        # Boundary loss mse
        yB = 1*tf.reduce_mean(tf.square(u_boundary_pred - u_boundary)) 
               
               
        # tf.print(yB)
        # tf.print(yS)
        
        
        # Total loss as the linear combination of the losses
        total_loss = yS + 100*yB 
        
                
        
        return total_loss, yS, yB
        
    
    
    def net_data_loss(self, x): 
        # Compute the data loss
        
        # Model prediction
        u = self.model(x)
        
        return  u
    
    def net_boundary_loss(self, x): 
        # Compute the boundary loss
        
        # Model prediction
        u = self.model(x)
        
        return  u
    
    
    def net_equation_loss(self):
        # Compute the pde loss
        nu = self.nu 
        x_colloc = self.x_t_colloc
    
        with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_colloc)
                
                # Model prediction
                u = self.model(x_colloc)
                
                # First order derivative wrt to [x,t]
                du = tape.gradient(u, x_colloc)

                u_t = du[:, 1]    # Slicing the gradient wrt time
                u_x = du[:, 0]    # Slicing the gradient wrt x
                
                # Second order derivative wrt [x,t]
                duu = tape.gradient(u_x, x_colloc)
                
                u_xx = duu[:, 0]  #  Slicing the gradient wrt x
                
        ## Conservation loss at each collocation point as 2d tensor array
        f = u_t + u*u_x - nu*u_xx

        return f
    

    
    # For the final prediction 
    def predict(self, x):
        
        u = self.model(x) 
    
        return u
 
    

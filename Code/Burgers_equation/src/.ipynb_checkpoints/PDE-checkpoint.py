#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:41:17 2022

@author: vikas
"""

import sys 
sys.path.insert(0, '../optimizer' )
import tensorflow as tf
import numpy as np
# from scipy.interpolate import griddata
# import numpy as np
#import tensorflow_probability as tfp
from NeuralNet_optimizer import Seq_NN, PhysicsInformedNN


#%%

class neural_net(Seq_NN):
    
    def __init__(self, ub, lb, layers):
        super(neural_net, self).__init__(ub, lb, layers)
    
    def call(self, inputs, training=False):
        output = super(neural_net, self).call(inputs)
        
        return output[:,0]
    
    
class PINN_PDE(PhysicsInformedNN):
    
    def __init__(self, #x0, y0, theta0, u0, v0, 
                 x_t_boundary,
                 u_boundary,
                 x_t_colloc,
                 x_t_sim,
                 u_sim,
                 ub, lb, 
                 layers,
                 scale,
                 scale_out,
                 nu
                 ):
        
        super(PINN_PDE, self).__init__()
        
        # Network architectures
        
        self.model = neural_net(ub, lb, layers)
        
        # Data initialization
        
        self.scale_max = scale[0]
        self.scale_min = scale[1]
        self.scale_out = scale_out
        self.x_t_boundary = x_t_boundary
        self.u_boundary = u_boundary
        self.x_t_colloc = x_t_colloc
        self.x_t_sim = x_t_sim
        self.u_sim = u_sim
        self.ub = ub[0]
        self.lb = lb[0]
        self.layers = layers
        self.nu = nu
        
        # self.theta_map_matrix = tf.reshape(theta_map, [2001,1])*tf.ones((1,len(x_f)), dtype=tf.float64)

        
        
    def loss(self):
        

        u_boundary = self.u_boundary                  
        u_sim = self.u_sim

                
        # We use only two equations here (mass and progress variable)
        progress_loss = self.net_progress_loss()
        # momentum_loss1 = self.net_momentum_loss1()
        # momentum_loss2 = self.net_momentum_loss2()      
        #progress_loss = self.net_progress_loss() 
                
        
        # PDE losses
        yS = 1*tf.reduce_mean(tf.square(progress_loss)) #+ \
                #1*tf.reduce_mean(tf.square(progress_loss)) #+\
                # 1*tf.reduce_mean(tf.square(momentum_loss1)) #+ \
                    # 0*tf.reduce_mean(tf.square(momentum_loss2)) #+ \
        
        
        # Loss from boundary conditions (in -1 to 1 scale, since tanh activation function)
        u_boundary_pred = self.net_data_loss(self.x_t_boundary)
        

    
        yB = 1*tf.reduce_mean(tf.square(u_boundary_pred - u_boundary)) 
               
               
        # tf.print(yB)
        # tf.print(yS)
        
        # Predictions in the range -1 to 1 (since tanh activation functions used)
        u_sim_pred = self.net_data_loss(self.x_t_sim)
        
        # Data loss
        yD = tf.reduce_mean(tf.square(u_sim_pred- u_sim))  #+\
            # 0*tf.reduce_mean(tf.square(u_data_pred - u_data_scaled)) #+ \
                 # tf.reduce_mean(tf.square(v_data_pred - V_data)) 
        
        
        total_loss = 1*yS + 1e3*yB + 1e4*yD # Scaling factor for each loss term (sensitive parameter)
        
        
        ## Losses which are not computed are set to 0 (since their values
        ## are needed to be saved in the framework)
        
        mass_loss = 0
        #progress_loss = 0
        momentum_loss1 = 0   # Momentum in x direction loss
        momentum_loss2 = 0   # Momentum in y direction loss
        YB = tf.constant(0)
        #yD = tf.constant(0)
        yI = tf.constant(0)  # Initial condition loss
        yN = tf.constant(0)  # Neumann loss
        yW = tf.constant(0)  # Reaction rate loss
        
        return total_loss, tf.reduce_mean(tf.square(mass_loss)), tf.reduce_mean(tf.square(progress_loss)), \
    tf.reduce_mean(tf.square(momentum_loss1)), tf.reduce_mean(tf.square(momentum_loss2)), yB, yN, yI, yD, yW
        

    def net_progress_loss(self):
        nu = self.nu 
        x_colloc = self.x_t_colloc
    
        with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_colloc)
        
                u1 = self.model(x_colloc)
                u = (u1 + self.scale_out[1]) * (self.scale_max[0] - self.scale_min[0]) / self.scale_out[0] + self.scale_min[0]
                du = tape.gradient(u, x_colloc)
                u_t = du[:, 1:]
                u_x = du[:, 0:1]
                duu = tape.gradient(u_x, x_colloc)
                u_xx = duu[:, 0:1]

        f = u_t + u * u_x - nu * u_xx

        return f

    def net_data_loss(self, x): 
        # Compute the data loss
        
        u1 = self.model(x)
        
        u = (u1 + self.scale_out[1])*(self.scale_max[0] - self.scale_min[0])/self.scale_out[0] + self.scale_min[0]

        return  u    

    # Needed by the Loss
    def net_mass_loss(self):
        # Mass conservation
        
        x_f = self.x_f
        
        RHO_IN = self.RHO_IN
        U_IN = self.U_IN
        alpha = self.alpha
        
        with tf.GradientTape(persistent=False) as tape:    
            tape.watch(x_f)
            
            theta1, u1 = self.model(x_f)          # Predictions between -1 and 1  
                        
            # Rescale to values corresponding to non-dimensional form
            theta = (theta1 + self.scale_out[1])/self.scale_out[0]  
            u = (u1 + self.scale_out[1])*(self.scale_max[0] - self.scale_min[0])/self.scale_out[0] + self.scale_min[0]
            
            rho = 1/(1 + alpha*theta/(1-alpha))
            rhoU = rho*u
            
        rhoU_x = tape.gradient(rhoU, x_f)
                            
        # u_t = tape.gradient(u, t_f)
        # v_t = tape.gradient(v, t_f)        
            
        # u_xx = tape.gradient(u_x, x_f)
        # v_xx = tape.gradient(v_x, x_f)
    
        del tape
    
        mass_loss = rhoU_x
        # tf.print(rhoU)
        # mass_loss1 = rhoU*U_IN*RHO_IN - RHO_IN*U_IN
    
        return mass_loss
    

    
    # For the final prediction 
    def predict(self, x):
        
        u1 = self.model(x) 
        
        # RHO_IN = self.RHO_IN
        # alpha = self.alpha
               
        u = (u1 + self.scale_out[1])*(self.scale_max[0] - self.scale_min[0])/self.scale_out[0] + self.scale_min[0]
    
    
        return u
 
    

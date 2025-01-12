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
# from scipy.interpolate import griddata
# import numpy as np
import tensorflow_probability as tfp
from NeuralNet_optimizer import Seq_NN, PhysicsInformedNN


#%%

class neural_net(Seq_NN):
    
    def __init_(self, ub, lb, layers, hid_activation, out_activation):
        super(neural_net, self).__init__(ub, lb, layers, hid_activation, out_activation)
    
    def call(self, inputs, training=False):
        output = super(neural_net, self).call(inputs)
        
        return output[:,0], output[:,1]
    
    
class PINN_PDE(PhysicsInformedNN):
    
    '''
    The current class incorporates the data, boundary, and PDE losses.
    
    In the current example, we are solving a 1D flame problem using two equations:
        Equation 1 is the mass conservation 
        Equation 2 is the progress variable conservation
        
        We use the non-dimensional form of the equations
        
    The boundaries are the two points on the left and the right of the domain
    
    The data loss will include data points you would like the model to learn
    
    At the end, the total loss is evaluated as the linear combination of the losses
    
    '''
    
    def __init__(self, 
                 hid_activation,
                 out_activation,
                 x_inlet, x_outlet, 
                 theta_inlet, u_inlet,
                 theta_outlet, u_outlet,
                 x_f, 
                 X_data, 
                 ub, lb, 
                 xmax, U_IN,
                 layers,
                 constants,
                 scale,
                 scale_out,
                 theta_data,
                 u_data,
                 rho_data,
                 RePr,
                 wdot_map,
                 theta_map):
        
        super(PINN_PDE, self).__init__()
        
        # Network architectures
        
        self.model = neural_net(ub, lb, layers, hid_activation, out_activation)
        
        # Data initialization
        
        self.RHO_IN = constants[0]
        self.alpha = constants[1]
        self.lambda_by_CP = constants[2]
        
        self.scale_max = scale[0]
        self.scale_min = scale[1]
        self.scale_out = scale_out
                
        self.theta_data = theta_data
        self.u_data = u_data
        
        self.x_inlet = x_inlet
        self.x_outlet = x_outlet
                
        self.theta_inlet = theta_inlet
        self.u_inlet = u_inlet
        self.theta_outlet = theta_outlet
        self.u_outlet = u_outlet
        
        self.x_f = x_f
        self.X_data = X_data
        
        self.xmax = xmax
        self.U_IN = U_IN
        
        self.RePr = RePr
        
        self.wdot_map = wdot_map
        self.theta_map =theta_map
        
        
        # self.theta_map_matrix = tf.reshape(theta_map, [2001,1])*tf.ones((1,len(x_f)), dtype=tf.float64)

        
        
    def loss(self):
        
        RHO_IN = self.RHO_IN
        alpha = self.alpha
        U_IN = self.U_IN
        
        theta_inlet = self.theta_inlet
        u_inlet = self.u_inlet
        
        
        theta_outlet = self.theta_outlet
        u_outlet = self.u_outlet
                
        
        u_data = self.u_data
        rho_data = 1/u_data # In non-dimensional form
        theta_data = (1-rho_data)*(1-self.alpha)/(rho_data*self.alpha)
                
        # We use only two equations here (mass and progress variable)
        mass_loss = self.net_mass_loss()   
        progress_loss = self.net_progress_loss() 
        
        
        # MSE of each loss
        mass_loss_mse = tf.reduce_mean(tf.square(mass_loss))
        progress_loss_mse = tf.reduce_mean(tf.squ1are(progress_loss))
                
        
        # PDE losses
        yS = 1*mass_loss_mse + 1*progress_loss_mse 
        
        
        # Loss from boundary conditions (in -1 to 1 scale, since tanh activation function)
        theta_inlet_pred, u_inlet_pred = self.net_boundary(self.x_inlet)
        theta_outlet_pred, u_outlet_pred = self.net_boundary(self.x_outlet)
        
        ######
        # Scaling the non-dimensional form values to -1 to 1 scale
        theta_inlet_scaled = theta_inlet*self.scale_out[0] - self.scale_out[1]
        theta_outlet_scaled = theta_outlet*self.scale_out[0] - self.scale_out[1]
        u_inlet_scaled = ((u_inlet - self.scale_min[0])/(self.scale_max[0] - 
                                                      self.scale_min[0]))*self.scale_out[0] - self.scale_out[1]         
        u_outlet_scaled = ((u_outlet - self.scale_min[0])/(self.scale_max[0] - 
                                                      self.scale_min[0]))*self.scale_out[0] - self.scale_out[1] 
        
        #######  Boundary loss 
    
        yB = 1*tf.reduce_mean(tf.square(theta_inlet_pred - theta_inlet_scaled)) + \
               1*tf.reduce_mean(tf.square(u_inlet_pred - u_inlet_scaled)) + \
               1*tf.reduce_mean(tf.square(theta_outlet_pred - theta_outlet_scaled)) +\
                    1*tf.reduce_mean(tf.square(u_outlet_pred - u_outlet_scaled)) 
               
        # tf.print(yB)
        # tf.print(yS)
        
        # Predictions in the range -1 to 1 (since tanh activation functions used)
        theta_data_pred, u_data_pred = self.net_data_loss(self.X_data)
        
        # Scaling the ground truth from non-dimensional form to (-1 to 1)        
        theta_data_scaled = theta_data*self.scale_out[0] - self.scale_out[1]
        u_data_scaled = ((u_data - self.scale_min[0])/(self.scale_max[0] - 
                                                      self.scale_min[0]))*self.scale_out[0] - self.scale_out[1]  
        
        
        # Data loss  (only theta fields used for data loss)
        yD = tf.reduce_mean(tf.square(theta_data_pred - theta_data_scaled))  #+\
            # 0*tf.reduce_mean(tf.square(u_data_pred - u_data_scaled))
        
        ## Linear combunation of all the losses
        total_loss = 1*yS + 1e2*yB + 1e2*yD # Scaling factor for each loss term (sensitive parameter)
        
        # REMEMBER to return total loss as the first argument (needed in optimizer),
        # The other values can be returned in any order (just make sure to have them saved in your main file accordingly)
        
        return total_loss, mass_loss_mse, progress_loss_mse, yB, yD
        


    # Data loss
    def net_data_loss(self, x): 
        # Compute the data loss
        
        theta1, u1 = self.model(x)

        return theta1, u1
    
        
    # Boundary loss
    def net_boundary(self, x):
        # Compute the boundary loss
        
        theta1, u1 = self.model(x)

        return theta1, u1

    # Mass conservation loss
    def net_mass_loss(self, x=None):
        # x=None gives the possibilty of evaluating models performance later
        # This can be done by providing any random spatial coordinates 
        # and checking if the mass conservation is satisfied

        if x == None:
            # Collocation points provided in the argument of this class
            x_f = self.x_f
        else:
            # External collocation points for later evaluation
            x_f = x
        
        
        RHO_IN = self.RHO_IN
        U_IN = self.U_IN
        alpha = self.alpha
        
        with tf.GradientTape(persistent=False) as tape:    
            tape.watch(x_f)
            
            theta1, u1 = self.model(x_f)          # Predictions between -1 and 1  
                        
            # Rescale to values corresponding to the non-dimensional form
            theta = (theta1 + self.scale_out[1])/self.scale_out[0]  
            u = (u1 + self.scale_out[1])*(self.scale_max[0] - self.scale_min[0])/self.scale_out[0] + self.scale_min[0]
            
            rho = 1/(1 + alpha*theta/(1-alpha))
            rhoU = rho*u
            
        rhoU_x = tape.gradient(rhoU, x_f)
                            
    
        del tape
    
        mass_loss = 1*rhoU_x
    
        return mass_loss
    
    
   
    # Progress variable conservation loss
    def net_progress_loss(self, x=None):
        
        # x=None gives the possibilty of evaluating models performance later
        # This can be done by providing any random spatial coordinates 
        # and checking if the mass conservation is satisfied
        if x == None:
            x_f = self.x_f
        else:
            x_f = x
            
        alpha = self.alpha
        RePr = self.RePr
        theta_map = self.theta_map
        wdot_map = self.wdot_map
        xmax = self.xmax
        U_IN = self.U_IN
        
        
        with tf.GradientTape(persistent=True) as tape:    
            
            tape.watch(x_f)
            theta1, u1 = self.model(x_f)
            
            theta = (theta1 + self.scale_out[1])/self.scale_out[0]
            u = (u1 + self.scale_out[1])*(self.scale_max[0] - self.scale_min[0])/self.scale_out[0] + self.scale_min[0]
            
            rho = 1/(1 + alpha*theta/(1-alpha))
            rhoUtheta = rho*u*theta
            # rhoU = rho*u
            theta_x = tape.gradient(theta, x_f)
        
        rhoUtheta_x = tape.gradient(rhoUtheta, x_f)
        theta_xx = tape.gradient(theta_x, x_f)
        
        # For interpolation in tensorflow
        wdot = tfp.math.batch_interp_rectilinear_nd_grid(x=tf.reshape(theta,[-1,1]), 
                                                             x_grid_points=(theta_map,),
                                                             y_ref = wdot_map, axis=-1)
        
        del tape
        
        loss1 = rhoUtheta_x
        loss2 = theta_xx/(RePr)
        
        progress_loss = loss1 - loss2 - wdot*xmax/U_IN
        
        
        return progress_loss
   
    
    # For the final prediction 
    def predict(self, x):
        
        theta1, u1 = self.model(x) 
        
        # RHO_IN = self.RHO_IN
        # alpha = self.alpha
        
        # This scales the prediction from (-1, 1) to the non-dimensionalized form 
        theta = (theta1 + self.scale_out[1])/self.scale_out[0]            
        u = (u1 + self.scale_out[1])*(self.scale_max[0] - self.scale_min[0])/self.scale_out[0] + self.scale_min[0]
    
    
        return theta, u
 
    

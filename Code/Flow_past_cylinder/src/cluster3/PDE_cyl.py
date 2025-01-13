#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:41:17 2022

@author: vikas
"""

import sys
sys.path.insert(0, '../../optimizer')
import tensorflow as tf
import numpy as np
# from scipy.interpolate import griddata
# import numpy as np
# import tensorflow_probability as tfp --> tf.__verion__ 2.13, 2.15
from NeuralNet_optimizer import Seq_NN, PhysicsInformedNN


# %%

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
                 # x_l, x_L,  # Boundary locations in the x-direction
                 # u_l, v_l, u_L, v_L,  # Boundary velocities at x_l and x_L
                 # y_b, y_B,  # Boundary locations in the y-direction
                 # u_b, v_b, u_B, v_B,  # Boundary velocities at y_b and y_B
                 x_boundary, y_boundary,
                 u_bc, v_bc,  # Boundary conditions
                 x_cylinder, y_cylinder,  # Cylinder points
                 u_cylinder, v_cylinder,  # Cylinder velocity conditions
                 gamma,
                 x_f, y_f,
                 x_data, y_data,  # Data points
                 u_data, v_data,  # Observed velocity data
                 ub, lb,  # Upper and lower bounds for normalisation
                 # constants,  # Physical constants
                 Re,
                 scale, scale_out,  # Scaling information
                 layers  # Neural network layers
                 ):

        super(PINN_PDE, self).__init__()

        # Network architectures
        self.model = neural_net(ub, lb, layers, hid_activation, out_activation)

        ##### Data initialization
        # Scaling parameters
        self.scale_max = scale[0]
        self.scale_min = scale[1]
        self.scale_out = scale_out

        # Domain Boundary data
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
        self.u_bc = u_bc
        self.v_bc = v_bc

        # Cylinder boundary data
        self.x_cylinder = x_cylinder
        self.y_cylinder = y_cylinder
        self.u_cylinder = u_cylinder
        self.v_cylinder = v_cylinder

        # Collocation points
        self.x_f = x_f
        self.y_f = y_f

        # Data points
        self.x_data = x_data
        self.y_data = y_data
        self.u_data = u_data
        self.v_data = v_data

        # constants
        self.Re = Re


    def loss(self):
        """
        Calculate the total loss for the PINN: includes physical loss, boundary loss, cylinder loss, and data loss.
        """

        # # Physical loss (mass and momentum conservation)
        loss_mass, loss_momx, loss_momy = self.net_physical_loss()
        # loss_mass = self.net_physical_loss()
        yS = loss_momx + loss_momy

        # # Boundary loss
        yB = self.net_boundary_loss()

        # Cylinder boundary loss
        yC = self.net_cylinder_loss()

        # Data loss
        yD = self.net_data_loss()

        # Total loss
        total_loss = 10*yB + 10*yC + 100*yD + yS
        # return total_loss, loss_mass, loss_momx, loss_momy, yB, yC, yD # return all of them to further analyse convergence behaviour of different losses
    
        # total_loss = 10*yB + 10*yC + 100*yD + yS
        return total_loss, loss_momx, loss_momy, yB, yC, yD, yS  # return all of them to further analyse convergence behaviour of different losses

    # # Data loss
    def net_data_loss(self): 
        """
        Computes the data loss using observed velocity data at given points.
        """

        # Predict u, v at data points
        u_pred, v_pred = self.model(tf.stack([self.x_data, self.y_data], axis=1))

        # Compute MSE for data points
        loss_data = tf.reduce_mean(tf.square(u_pred - self.u_data)) + tf.reduce_mean(tf.square(v_pred - self.v_data))
        return loss_data


    def net_cylinder_loss(self):
        """
        Compute the cylinder loss enforcing the no-slip condition at the cylinder surface
        """
        
        # self.x_cylinder = tf.reshape(self.x_cylinder, [-1, 1])  # Reshape to (N, 1)
        # self.y_cylinder = tf.reshape(self.y_cylinder, [-1, 1])  # Reshape to (N, 1)
        
        # Predict u, v at cylinder boundary points
        u_pred, v_pred = self.model(tf.stack([self.x_cylinder, self.y_cylinder], axis=1))
        
        # Compute MSE for cylinder boundary conditions
        loss_cylinder = tf.reduce_mean(tf.square(u_pred - self.u_cylinder)) + tf.reduce_mean(tf.square(v_pred - self.v_cylinder))
        return loss_cylinder


    # Boundary loss
    def net_boundary_loss(self):
        """
        Compute the boundary loss using sampled boundary points
        """
        
        # self.x_boundary = tf.reshape(self.x_boundary, [-1, 1])  # Reshape to (N, 1)
        # self.y_boundary = tf.reshape(self.y_boundary, [-1, 1])  # Reshape to (N, 1)
        
        # Predict u, v at boundary points
        u_pred, v_pred = self.model(tf.stack([self.x_boundary, self.y_boundary], axis=1))

        # Compute MSE for boundary conditions
        loss_boundary = tf.reduce_mean(tf.square(u_pred - self.u_bc)) + tf.reduce_mean(tf.square(v_pred - self.v_bc))
        return loss_boundary


    def net_physical_loss(self):
        """
        Computes the physical loss based on conservation equations
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_f)
            tape.watch(self.y_f)
            
            # self.x_f = tf.reshape(self.x_f, [-1, 1])  # Reshape to (N, 1)
            # self.y_f = tf.reshape(self.y_f, [-1, 1])  # Reshape to (N, 1)

            # Predict u, v
            u, v = self.model(tf.stack([self.x_f, self.y_f], axis=1))

            # Compute first derivatives
            u_x = tape.gradient(u, self.x_f)
            u_y = tape.gradient(u, self.y_f)
            v_x = tape.gradient(v, self.x_f)
            v_y = tape.gradient(v, self.y_f)

        # # Compute second derivatives
        # u_xx = tape.gradient(u_x, self.x_f)
        # u_yy = tape.gradient(u_y, self.y_f)
        # v_xx = tape.gradient(v_x, self.x_f)
        # v_yy = tape.gradient(v_y, self.y_f)
            
        del tape
        
        # Compute physical losses
        p_x = 0  # incompressable fluid at constant pressure
        p_y = 0
        # rho eliminated

        l_mass = u_x + v_y  # Mass conservation
        # l_momx = u * u_x + v * u_y + p_x - (1 / self.Re) * (u_xx + u_yy)  # x-momentum
        # l_momy = u * v_x + v * v_y + p_y - (1 / self.Re) * (v_xx + v_yy)  # y-momentum

        # Compute MSE for physical losses
        loss_mass = tf.reduce_mean(tf.square(l_mass))
        # loss_momx = tf.reduce_mean(tf.square(l_momx))
        # loss_momy = tf.reduce_mean(tf.square(l_momy))

        # Total physical loss
        # return loss_mass, loss_momx, loss_momy
        return loss_mass # , loss_momx, loss_momy


    # For the final prediction
    def predict(self, x, y):
        """
        Predict u and v for given x and y coordinates.
        """
        u_pred, v_pred = self.model(tf.stack([x, y], axis=1))
        return u_pred, v_pred



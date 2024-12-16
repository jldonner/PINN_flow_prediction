#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:46:52 2022

@author: vikas
"""

import tensorflow as tf
import scipy.optimize 

import time
from tqdm import tqdm
import numpy as np

#%% 

class Seq_NN(tf.keras.Sequential):
    
    def __init__(self, ub, lb, layers):
        super(Seq_NN, self).__init__()
        tf.keras.backend.set_floatx('float64')
        self.t_last_callback = 0
        
        self.lb = lb
        self.ub = ub
        
        self.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        
        #self.add(tf.keras.layers.Lambda(
           # lambda X: 2.0*(X-self.lb)/(self.ub - self.lb) - 1.0))
            # lambda X: 1.0*(X)))
        
        for width in layers[1:-1]:
            self.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer=tf.keras.initializers.GlorotUniform()))
            
        self.add(tf.keras.layers.Dense(
                layers[-1], activation=tf.nn.tanh,
                kernel_initializer=tf.keras.initializers.GlorotUniform()))
        
        
        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i in range(len(layers)-1):
            # if i != 1:
            self.sizes_w.append(int(layers[i] * layers[i+1]))
            self.sizes_b.append(int(layers[i+1]))
            
                
    def get_weights(self, convert_to_tensor=True):
        
        w = []
        for layer in self.layers[1:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        if convert_to_tensor:
            w = tf.convert_to_tensor(w)
        
        # Make the output Fortran contiguous
        w = np.copy(w, order='F')
        
        return w
    
    def set_weights(self, w):
        
        for i, layer in enumerate(self.layers[1:]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i+1])
                        
            w_b = w[start_weights:end_weights]
            
            weights = w_b[0:self.sizes_w[i]]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            
            
            biases = w_b[self.sizes_w[i]:]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)
                
                
                
class PhysicsInformedNN():
    
    def __init__(self):
        pass
   
    def loss_and_flat_grad(self, w):
        
        with tf.GradientTape() as tape:
            self.model.set_weights(w)
            loss_value, _, _, _, _, _, _, _, _, _ = self.loss()
        grad = tape.gradient(loss_value, self.model.trainable_variables)
        grad_flat = []
        for g in grad:
            grad_flat.append(tf.reshape(g, [-1]))
        grad_flat = tf.concat(grad_flat, 0)
        
        # Make the output Fortran contiguous
        loss_value = np.copy(loss_value, order='F')
        grad_flat = np.copy(grad_flat, order='F')
        # print(mass_loss)
        return loss_value, grad_flat

    def loss(self):
        pass
    
    
    def train(self, Adam_iterations, LBFGS_max_iterations, checkpoint_str, lr_epochs, lr_list):
        
        # ADAM training
        self.Adam_iter = Adam_iterations
        self.LBFGS_iter = LBFGS_max_iterations
        self.checkpoint_str = checkpoint_str
        self.Adam_hist = []
        self.mass_cons = []
        self.progress_cons = []
        self.momentum_cons1 = []
        self.momentum_cons2 = []
        self.bound_loss = []
        self.neu_loss = []
        self.initial_loss = []
        self.data_loss = []
        self.wdot_loss = []
        if (Adam_iterations):
            
            print('~~ Adam optimization ~~')
            
            lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_epochs, lr_list)
            
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
            
            start_time = time.time()   
            iteration_start_time = start_time
            #Train step
            for i in range(Adam_iterations):
                current_loss, mass_loss, progress_loss, mom_loss1, mom_loss2, b_loss, n_loss, i_loss, d_loss, w_loss = self.Adam_train_step(optimizer)
                if i > 0:
                    if current_loss.numpy() < np.min(self.Adam_hist): 
                        self.model.save_weights(checkpoint_str)
                        
                    # if (i+1) % 50 == 0: 
                    #     self.model.save_weights(checkpoint_str + '_' + str(i+1))
                        
                self.Adam_hist.append(current_loss.numpy())
                self.mass_cons.append(mass_loss.numpy())
                self.progress_cons.append(progress_loss.numpy())
                self.momentum_cons1.append(mom_loss1.numpy())
                self.momentum_cons2.append(mom_loss2.numpy())
                self.initial_loss.append(i_loss.numpy())
                self.bound_loss.append(b_loss.numpy())
                self.neu_loss.append(n_loss.numpy())
                self.data_loss.append(d_loss.numpy())
                self.wdot_loss.append(w_loss.numpy())
                
                iteration_time = str(time.time()-iteration_start_time)[:5]
                print('Loss:', current_loss.numpy(), 'time:', iteration_time, 'iter: '+str(i)+'/'+str(Adam_iterations) )                
                iteration_start_time = time.time()
                
                # print(current_loss.numpy(), np.min(self.Adam_hist))
                # print(current_loss.numpy() < np.min(self.Adam_hist))
                
                
            elapsed = time.time() - start_time   
            # total_loss, mass_loss, progress_loss = self.loss()
            # self.Adam_hist.append(total_loss.numpy())
            # self.mass_cons.append(mass_loss.numpy())
            # self.progress_cons.append(progress_loss.numpy())
            print('Training time: %.4f' % (elapsed))
            
        self.lbfgs_iter = 0
        # LBFGS trainig        
        total_loss, mass_loss, progress_loss, mom_loss1, mom_loss2, b_loss, n_loss, i_loss, d_loss, w_loss = self.loss()
        self.LBFGS_hist = [total_loss.numpy()]
        self.mass_cons.append(mass_loss.numpy())
        self.progress_cons.append(progress_loss.numpy())
        self.momentum_cons1.append(mom_loss1.numpy())
        self.momentum_cons2.append(mom_loss2.numpy())
        self.bound_loss.append(b_loss.numpy())
        self.neu_loss.append(n_loss.numpy())
        self.initial_loss.append(i_loss.numpy())
        self.data_loss.append(d_loss.numpy())
        self.wdot_loss.append(w_loss.numpy())
        
        if (LBFGS_max_iterations):
            
            print('~~ L-BFGS optimization ~~')
                        
            maxiter = LBFGS_max_iterations
            self.t_last_callback = time.time()
            
            results = scipy.optimize.minimize(self.loss_and_flat_grad,
                                              self.model.get_weights(),
                                              method='L-BFGS-B',
                                              jac=True,
                                              callback=self.callback,
                                              options = {'maxiter': maxiter,
                                                         'maxfun':500000,
                                                         'maxcor': 50,
                                                         'maxls': 50,
                                                         'ftol' : 1.0 * np.finfo(float).eps})
            
            optimal_w = results.x 
            self.model.set_weights(optimal_w)
            
            total_loss, mass_loss, progress_loss, mom_loss1, mom_loss2, b_loss, n_loss, i_loss, d_loss, w_loss = self.loss()
    
            print('~~ model trained ~~','\n','Final loss:',total_loss.numpy())
    
        return self.Adam_hist, self.LBFGS_hist
    
    @tf.function
    def Adam_train_step(self, optimizer):
        
        with tf.GradientTape() as tape:
            total_loss, mass_loss, progress_loss, mom_loss1, mom_loss2, b_loss, n_loss, i_loss, d_loss, w_loss = self.loss()
 
        grads = tape.gradient(total_loss, self.model.trainable_variables)    
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
  
        return total_loss, mass_loss, progress_loss, mom_loss1, mom_loss2, b_loss, n_loss, i_loss, d_loss, w_loss

    def callback(self, pars):
        self.lbfgs_iter = self.lbfgs_iter + 1
        checkpoint_str = self.checkpoint_str
        t_interval = str(time.time()-self.t_last_callback)[:5]
        
        total_loss, mass_loss, progress_loss, mom_loss1, mom_loss2, b_loss, n_loss, i_loss, d_loss, w_loss = self.loss()
        loss_value = total_loss.numpy()
        
        if total_loss.numpy() < np.min(self.LBFGS_hist): 
            self.model.save_weights(checkpoint_str)
            
        # if (self.lbfgs_iter + 1) % 50 == 0: 
        #     self.model.save_weights(checkpoint_str + '_' + str(self.lbfgs_iter + self.Adam_iter + 1))
            
        self.LBFGS_hist.append(loss_value)
        self.mass_cons.append(mass_loss.numpy())
        self.progress_cons.append(progress_loss.numpy())
        self.momentum_cons1.append(mom_loss1.numpy())
        self.momentum_cons2.append(mom_loss2.numpy())
        self.bound_loss.append(b_loss.numpy())
        self.neu_loss.append(n_loss.numpy())
        self.initial_loss.append(i_loss.numpy())
        self.data_loss.append(d_loss.numpy())
        self.wdot_loss.append(w_loss.numpy())
        
        print('Loss:', loss_value, 'time:', t_interval, 'iteration:', self.lbfgs_iter)
        self.t_last_callback = time.time()
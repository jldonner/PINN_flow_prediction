#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:54:08 2023

@author: vikas
"""

import numpy as np 
import tensorflow as tf
from scipy import io as sio
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.interpolate import griddata
import h5py
from matplotlib.ticker import FormatStrFormatter
from scipy import interpolate
from PDE import PINN_PDE

#%% Constants


# Read the data from reference file
hf = h5py.File('../Data/burgers_shock.h5','r')
t = np.array(hf.get('/t')).T
x = np.array(hf.get('/x')).T
Exact = np.array(hf.get('/usol'))
hf.close()
nu = 0.01/np.pi

X, T = np.meshgrid(x,t)
X_all = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
Uexact_all = Exact.flatten()[:,None]              

# Domain bounds
lb = X_all.min(0)
ub = X_all.max(0)


# First, we define the points associated with the initial condition, which is known
# Uniformly distributed points in x-domain [-1,1] for the initial condition
# u = -sin(pi*x)

Nx_init = 50
x_init = np.linspace(lb[0],ub[0],Nx_init)

u_x_init = -1*np.sin(np.pi*x_init)

X_init = np.hstack((x_init.reshape(-1,1), np.zeros(len(x_init)).reshape(-1,1)))



# Second, we define the points associated with the left and right Dirichlet boundary conditions (u=0)
# Uniformly distributed points in t-domain [0,1] for the left BC

Nt_BC = 50

t_BC = np.linspace(lb[1], ub[1], Nt_BC)
u_train = np.zeros(len(t_BC))


T_init = np.hstack((-np.ones(len(t_BC)).reshape(-1,1),t_BC.reshape(-1,1) ))
#print(T_init)

# Uniformly distributed points in t-domain [0,1] for the right BC

T_init1 = np.hstack((np.ones(len(t_BC)).reshape(-1,1),t_BC.reshape(-1,1) ))
#print(T_init1)


# Stacking all the inputs and outputs
X_train = np.vstack((X_init, T_init, T_init1))
u_train = np.vstack((u_x_init.reshape(-1,1), u_train.reshape(-1,1), u_train.reshape(-1,1)))


#%% Creating collocation points
 
# Collocation points
N_colloc = 8000

# Sampled collocation points based on LHS
lhs = qmc.LatinHypercube(d=2, seed=1234)
X_colloc_train = lb + (ub-lb)*lhs.random(N_colloc)

#%% Convert to tensor

x_t_boundary = tf.convert_to_tensor(X_train)
u_boundary = tf.convert_to_tensor(u_train[:,0])    # Important to convert the shape from (-1,1) to (-1)
x_t_colloc = tf.convert_to_tensor(X_colloc_train)


#%% Buiilding the model


layers = [2, 20, 20, 20, 20, 20, 20, 20, 1]  

hid_activation = 'tanh'
out_activation = 'linear'


model = PINN_PDE(hid_activation,
                 out_activation,
                 x_t_boundary,
                 u_boundary,
                 x_t_colloc,
                 ub, lb, 
                 layers,
                 nu
                 )


#%%

# You can play around with these values
# You will observe that only Adam takes time to converge in comparison to a combination of Adam and LBFGS
adam_iterations = 300         # Number of training steps 
lbfgs_max_iterations = 1500     # Max iterations for lbfgs

# Below two variables are used for piecewise learning rate scheduler for ADAM
lr_list = [1e-3, 1e-4]    ## Learnig rate
lr_epochs = [200]        ## Epochs at which the learning rate should change


#%%

#### Training with ADAM followed by LBFGS
checkpoint_str = './Checkpoint/Burgers'
hist = model.train(adam_iterations, lbfgs_max_iterations, checkpoint_str, lr_epochs, lr_list)


#%%
######  Loss curve

plt.rc('font', family='serif')
plt.rc('font', size=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)


##### Plotting losses

fig, ax = plt.subplots(1,1, figsize=(9,4.5))



ax.semilogy(model.full_loss_list[0], linewidth=3, label='Total loss',
            color='#1f77b4')
ax.semilogy(model.full_loss_list[1], linewidth=3, label='PDE loss', 
            color='#ff7f0e')
ax.semilogy(model.full_loss_list[2], linewidth=3, label='Boundary loss',
            color='#8c564b') 



ax.set_xlabel('Epochs')
ax.set_ylabel('MSE')

# ax.set_ylim([1e-6, 1e9])

ax.legend(ncol=2)
plt.tight_layout()
plt.show()

#%% Loading trained weights (Will be useful when the models are trained on clusters)
# 
model.model.load_weights(checkpoint_str)

# model.model.load_weights('./Checkpoint/checkpoint_hydra')
# 
#%% Field prediction

u_pred = model.predict(X_all)

u_nn = np.reshape(u_pred,[100,256])
#print(t.shape)
plt.figure(figsize=(6,6))
plt.contourf(x[:,0],t[:,0],u_nn)

plt.show()






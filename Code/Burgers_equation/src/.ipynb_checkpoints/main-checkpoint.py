#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:54:08 2023

@author: vikas
"""

import numpy as np 
import tensorflow as tf
from scipy import io as sio
# from pyDOE import lhs
import matplotlib.pyplot as plt
import sys
from scipy.stats import qmc
from scipy.interpolate import griddata
import h5py
from matplotlib.ticker import FormatStrFormatter
from scipy import interpolate
from PDE import PINN_PDE
from Data_NN import Data_NN

#%% Constants




#umin = np.array([umin]) 
#umax = np.array([umax]) 

# Read the data from reference file
hf = h5py.File('./Data_Burgers/burgers_shock.h5','r')
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

scale_out = [2.0, 1.0]
umin = Exact.min()
umax = Exact.max()
scale= [umin,umax]

# First, we define the points associated with the initial condition, which is known
# Uniformly distributed points in x-domain [-1,1] for the initial condition
# u = -sin(pi*x)

Nx_init = 50
x_init = np.linspace(lb[0],ub[0],Nx_init)
x_init_scaled = x_init/ub[0]
#print(x_init)
u_x_init = -1*np.sin(np.pi*x_init)
u_x_init_scaled = u_x_init/umax
#print(u_x_init)

X_init = np.hstack((x_init.reshape(-1,1), np.zeros(len(x_init)).reshape(-1,1)))
X_init_scaled = np.hstack((x_init_scaled.reshape(-1,1), np.zeros(len(x_init)).reshape(-1,1)))



# Second, we define the points associated with the left and right Dirichlet boundary conditions (u=0)
# Uniformly distributed points in t-domain [0,1] for the left BC

Nt_BC = 50

t_BC = np.linspace(lb[1], ub[1], Nt_BC)
t_BC_scaled = t_BC/ub[1]
u_train = np.zeros(len(t_BC))
u_train_scaled = u_train/umax

x_scaled_init = -np.ones(len(t_BC))
x_scaled_init = x_scaled_init/ub[0]

T_init = np.hstack((-x_scaled_init.reshape(-1,1),t_BC_scaled.reshape(-1,1) ))
#print(T_init)

# Uniformly distributed points in t-domain [0,1] for the right BC

T_init1 = np.hstack((x_scaled_init.reshape(-1,1),t_BC_scaled.reshape(-1,1) ))
#print(T_init1)


# Stacking all the features

X_train = np.vstack((X_init_scaled, T_init, T_init1))
u_train = np.vstack((u_x_init_scaled.reshape(-1,1), u_train_scaled.reshape(-1,1), u_train_scaled.reshape(-1,1)))

import random


def zufallswerte(a, b, anzahl=3):
    
    zufallsindices = random.sample(range(len(a)), anzahl)
    zufallswerte_a = [a[i] for i in zufallsindices]
    zugehoerige_werte_b = [b[i] for i in zufallsindices]
    
    return zufallswerte_a, zugehoerige_werte_b


x_t_sim,u_sim = zufallswerte(X_all, Uexact_all, 500)
x_t_sim_scaled = x_t_sim/ub
u_sim_scaled = u_sim/umax

N_colloc = 4000

X_colloc_train = lb + (ub-lb)*lhs(2,N_colloc)
X_colloc_train_scaled = X_colloc_train/ub


layers = [2, 30, 30, 30, 30, 30, 30, 1]  

scale_max = tf.convert_to_tensor(np.array([umax, 2.]))
scale_min = tf.convert_to_tensor(np.array([umin, 0.]))
x_t_boundary = tf.convert_to_tensor(X_train)
u_boundary = tf.convert_to_tensor(u_train)
x_t_colloc = tf.convert_to_tensor(X_colloc_train_scaled)
x_t_sim = tf.convert_to_tensor(x_t_sim_scaled)
u_sim = tf.convert_to_tensor(u_sim_scaled)

model = PINN_PDE(x_t_boundary,
                 u_boundary,
                 x_t_colloc,
                 x_t_sim,
                 u_sim,
                 ub, lb, 
                 layers,
                 [scale_max, scale_min],
                 scale_out,
                 nu
                 )

#%%

adam_iterations = 500  # Number of training steps 
lbfgs_max_iterations = 10000 # Max iterations for lbfgs

lr_list = [1e-3,1e-4]
lr_epochs = [200]

# Loading the weights from the initialized fields for faster convergence
# model.model.load_weights(checkpoint_str_NN)

#%%

#### Training
checkpoint_str = './Checkpoint/1D_flame_theta_as_data_loss'
   
# model.model.load_weights('./Checkpoint/1D_theta_mp_1000_50000_nondim_30_6_init_10_10_100.index')
# model.model.load_weights(checkpoint_str_NN)


hist = model.train(adam_iterations, lbfgs_max_iterations, checkpoint_str, lr_epochs, lr_list)
# 

#%%

model.model.load_weights(checkpoint_str)
# 
#%% Field prediction

u_pred = model.predict(X_all)

u_pred = np.array(u_pred)*U_IN



# plt.plot(Input_all[:,0], u_pred)
# plt.plot(Input_all[:,0], u)

# plt.show()


#%%  plotting functions

plt.rc('font', family='serif')
plt.rc('font', size=28)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)

fig,ax = plt.subplots(1,1, figsize=(15,6))


ax1 = ax.twinx()
ax.plot(z,rho, label='Cantera density', linewidth=3, color='#cc0000')
# ax.plot(X_colloc[0:10],rho_pred[0:10], 'o', linewidth=3, color='#cc0000', markersize=10, fillstyle='none')
ax.plot(X_colloc[10:-10:10],rho_pred[10:-10:10], 'o', label='PINN density', 
        linewidth=3, color='#cc0000', markersize=10, fillstyle='none')
# ax.plot(z[-10:],rho_pred[-10:], 'o', linewidth=3, color='#cc0000', markersize=10, fillstyle='none')

ax.set_xlabel('Length')
ax.set_ylabel(r'Density $(kg/m³)$')
ax.legend(ncol=1,handleheight=1.4, labelspacing=0.0, handletextpad=0.2,
                borderpad=0.2, loc='best')
ax.set_xticks([0, 0.004, 0.008, 0.012, 0.016, 0.02])

ax1.plot(z,u, label='Cantera velocity', linewidth=3, color='#0000CD')
# ax1.plot(z[0:10],u_pred[0:10], 'd', linewidth=3, color='#228B22', markersize=10, fillstyle='none')
ax1.plot(X_colloc[10:-10:10],u_pred[10:-10:10], 'd', label='PINN velocity', linewidth=3, color='#0000CD',
          markersize=10, fillstyle='none')
# ax1.plot(z[-10:],u_pred[-10:], 'd', linewidth=3, color='#228B22', markersize=10, fillstyle='none')

ax1.set_ylabel(r'Velocity $(m/s)$')
# ax1.legend(ncol=1,handleheight=1.4, labelspacing=0.0, handletextpad=0.2,
#                 borderpad=0.2, loc='center right')

# ax1.plot(z,T, label='Cantera temperature', linewidth=3, color='#228B22')
# # ax1.plot(z[0:10],T_pred[0:10], 'd', linewidth=3, color='#228B22', markersize=10, fillstyle='none')
# ax1.plot(X_colloc[10:-10:10],T_pred[10:-10:10], 'd', label='PINN temperature', linewidth=3, color='#228B22',
#           markersize=10, fillstyle='none')
# # ax1.plot(z[-10:],T_pred[-10:], 'd', linewidth=3, color='#228B22', markersize=10, fillstyle='none')

# ax1.set_ylabel(r'Temperature $(K)$')
ax1.legend(ncol=1,handleheight=1.4, labelspacing=0.0, handletextpad=0.2,
                borderpad=0.2, loc='best')



plt.tight_layout()
plt.show()





#%%
######  Loss curve

plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)


##### Plotting losses

fig, ax = plt.subplots(1,1, figsize=(10,6))



ax.semilogy(model.Adam_hist + model.LBFGS_hist, linewidth=3, label='Total loss',
            color='#1f77b4')
ax.semilogy(model.mass_cons, linewidth=3, label='Mass cons loss', 
            color='#ff7f0e')

ax.semilogy(model.bound_loss, linewidth=3, label='Dirichlet loss',
            color='#9467bd')
ax.semilogy(model.progress_cons, linewidth=3, label='Progress loss',
            color='#8c564b') 
ax.semilogy(model.data_loss, linewidth=3, label='Data loss',
            color='#d62728')



ax.set_xlabel('Epochs')
ax.set_ylabel('MSE')

ax.set_ylim([1e-6, 1e9])

ax.legend(ncol=2)
plt.tight_layout()
plt.show()

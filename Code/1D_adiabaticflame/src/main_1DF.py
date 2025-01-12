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

#%% Constants

# Set data type

RHO_IN = np.array([1.1435])  # Inlet density 
U_IN = np.array([0.264])      # Inlet velocity
P_OUT = np.array([1e5])   # Outlet Pressure




#%% Read Cantera (simulation) data


z_dic = sio.loadmat('../Data/z.mat')
z = z_dic['z'].reshape(-1)

theta_dic = sio.loadmat('../Data/theta.mat')
theta = theta_dic['theta'].reshape(-1)

T_dic = sio.loadmat('../Data/T.mat')
T = T_dic['T'].reshape(-1)

u_dic = sio.loadmat('../Data/u.mat')
u = u_dic['u'].reshape(-1)

rho_dic = sio.loadmat('../Data/rho.mat')
rho = rho_dic['rho'].reshape(-1)

P_dic = sio.loadmat('../Data/P.mat')
P = P_dic['P'].reshape(-1)

Cond_dic = sio.loadmat('../Data/Cond.mat')
Cond = Cond_dic['Cond'].reshape(-1)

Visc_dic = sio.loadmat('../Data/Visc.mat')
Visc = Visc_dic['Visc'].reshape(-1)

CP_dic = sio.loadmat('../Data/CP.mat')
CP = CP_dic['CP'].reshape(-1)


#%% Bounds

# Set number of data points

N_r1 = 1000    # Coarse collocation points before the flame jump
N_r2 = 2000    # Fine collocation points in the flame regions
N_r3 = 1000    # Coarse collocation points after the flame jump

# Set boundary

xmin = 0.0     # Inlet x position
xmax = 0.02    # Outlet x position

## Assumed max and min values of the velocity
umax = 8.0
umin = -2.0

# Lower bounds
lb = np.array([xmin])   # Global Lower bound for time and x variables
# Upper bounds
ub = np.array([xmax])  # Global Upper bound for time and x variables

lb1 = np.array([0.0])    # Region before flame lower bound
ub1 = np.array([0.006])  # Region before flame upper bound

lb2 = np.array([0.006])  # Flame region lower bound
ub2 = np.array([0.010])  # Flame region upper bound

lb3 = np.array([0.010])  # Region after flame lower bound
ub3 = np.array([0.02])   # Region after flame upper bound


# This is used for scaling from (0,1) to (-1,1)
scale_out = [2.0, 1.0]


#%%

# Data points obtained from grids created after linear interpolation
data_points1 = np.linspace(xmin, xmax, 1000).reshape(-1,1)
data_points1_scaled = data_points1/ [xmax]

# Data points obtained by using the exactgrids from simulation mesh
data_points = z.reshape(-1,1)
data_points_scaled = data_points/ [xmax]

# Fields from the simulation (For using linearly interpolated grid fields use _grid_new)

U_data = u.reshape(-1,1)
T_data = T.reshape(-1,1)
theta_data = theta.reshape(-1,1)

alpha = (T.max() - T.min())/T.max()
rho_data = RHO_IN/(1 + alpha*theta_data/(1-alpha))


Y_grid = np.concatenate([theta_data, U_data], axis = 1)


#%%   Non-dimensionalizing the output values (dependent varibales)

T_data_scaled = T_data/T_data.min()
theta_data_scaled = 1*theta_data - 0
U_data_scaled = U_data/U_IN
rho_data_scaled = rho_data/RHO_IN

Y_grid_scaled = np.concatenate([theta_data_scaled, U_data_scaled], axis = 1)


#%%

lambda_by_CP = np.array([2.5e-5]) # Lambda/CP = Rho*D = Constant [Assumption]
T1 = np.array([T_data.min()])     # Inlet temperature
T2 = np.array([T_data.max()])    # Max temperature

alpha = (T2-T1)/T2

c1 = 1.71e-5
c2 = 110.4
mu = c1*(T_data**1.5/(T_data + c2))*((T1 + c2)/T1**1.5)

Re = RHO_IN*U_IN*xmax/mu
Pr = mu/lambda_by_CP

# scale_out = [2.0, 1.0]


#%% Generate samples for boundary conditions



#######
#######    theta, u, v and p at inlet
theta_inlet = np.zeros((1,1))         # theta at inlet
theta_inlet_scaled = 1*theta_inlet - 0

u_inlet = U_IN*np.ones((1,1))
u_inlet_scaled = u_inlet/U_IN               ### Non-dimensionalizing the velocity



##################
##################
Input_inlet = np.zeros((1,1))
Input_inlet_scaled = Input_inlet/xmax    ### Non-dimensionalizing the input

Output_inlet = np.concatenate([theta_inlet, u_inlet], axis=1)
Output_inlet_scaled = np.concatenate([theta_inlet_scaled, u_inlet_scaled], axis=1)



#%% Evaluate boundary condition at outlet

#########
#########    x and y at the outlet


#######
#######    theta, u, v and p at outlet

theta_outlet = 1.0*np.ones((1,1))          # theta at right
theta_outlet_scaled = 1*theta_outlet - 0

rho_outlet = RHO_IN/(1 + alpha*theta_outlet/(1-alpha))

u_outlet = U_IN*RHO_IN/rho_outlet    
u_outlet_scaled = u_outlet/U_IN            ### Non-dimensionalizing the velocity


##################
##################
Input_outlet = 0.02*np.ones((1,1))
Input_outlet_scaled = Input_outlet/xmax    ### Non-dimensionalizing the input

Output_outlet = np.concatenate([theta_outlet, u_outlet], axis=1)
Output_outlet_scaled = np.concatenate([theta_outlet_scaled, u_outlet_scaled], axis=1)

##################
##################


#%%


# Collect boundary and inital data in lists
X_bound = np.concatenate([Input_inlet, Input_outlet], axis=0)  # Collect the input features
X_bound_scaled = np.concatenate([Input_inlet_scaled, Input_outlet_scaled], axis=0)  # Collect the input features

Y_bound = np.concatenate([Output_inlet, Output_outlet], axis=0)  # Collect the output 
Y_bound_scaled = np.concatenate([Output_inlet_scaled, Output_outlet_scaled], axis=0)  # Collect the output 

#%% Collocation points


# Sampled collocation points based on LHS
lhs1 = qmc.LatinHypercube(d=1, seed=1234)
Colloc_train1 = lb1 + (ub1-lb1)*lhs1.random(N_r1)   #  
Colloc_train1_scaled = (Colloc_train1)/[xmax]
# 
lhs2 = qmc.LatinHypercube(d=1, seed=1234)
Colloc_train2 = lb2 + (ub2-lb2)*lhs2.random(N_r2)    #  0.006 to 0.01 (Possible region of flame)
Colloc_train2_scaled = (Colloc_train2)/[xmax]

lhs3 = qmc.LatinHypercube(d=1, seed=1234)
Colloc_train3 = lb3 + (ub3-lb3)*lhs3.random(N_r3)
Colloc_train3_scaled = (Colloc_train3)/[xmax]

X_colloc = np.concatenate([Colloc_train1, Colloc_train2, Colloc_train3], axis=0)
X_colloc = np.sort(X_colloc, axis=0)
X_colloc_scaled = X_colloc/xmax


#%% Concatenating data points from Cantera (simulation) and Boundary points 

X_data_all = np.concatenate([X_bound, data_points], axis=0)
X_data_all_scaled = np.concatenate([X_bound_scaled, data_points_scaled], axis=0)
                            

Y_data_all = np.concatenate([Y_bound,Y_grid], axis=0)
Y_data_all_scaled = np.concatenate([Y_bound_scaled, Y_grid_scaled], axis=0)



#%% Coversion to tensorflow

# Important to convert all the variables shape from (-1,1) to (-1) 

x_inlet = tf.convert_to_tensor(Input_inlet_scaled[:,0])
x_outlet = tf.convert_to_tensor(Input_outlet_scaled[:,0])

theta_inlet = tf.convert_to_tensor(Output_inlet_scaled[:,0])
u_inlet = tf.convert_to_tensor(Output_inlet_scaled[:,1])

theta_outlet = tf.convert_to_tensor(Output_outlet_scaled[:,0])
u_outlet = tf.convert_to_tensor(Output_outlet_scaled[:,1])


x_f = tf.convert_to_tensor(X_colloc_scaled[:,0])
x_f_grid = tf.convert_to_tensor(data_points1_scaled[:,0])

X_data = tf.convert_to_tensor(data_points_scaled[:,0]) 

U_data_tensor = tf.convert_to_tensor(U_data_scaled[:,0]) 
rho_data_tensor = tf.convert_to_tensor(rho_data_scaled[:,0]) 
# rhoU_data_tensor = tf.convert_to_tensor(rhoU_data[:,0])
theta_data_tensor = tf.convert_to_tensor(theta_data_scaled[:,0]) 

RHO_IN = tf.convert_to_tensor(RHO_IN)
alpha = tf.convert_to_tensor(alpha)
lambda_by_CP = tf.convert_to_tensor(lambda_by_CP)

scale_max = tf.convert_to_tensor(np.array([umax, 2.]))
scale_min = tf.convert_to_tensor(np.array([umin, 0.]))

Re_tensor = tf.convert_to_tensor(Re)
Pr_tensor = tf.convert_to_tensor(Pr)
RePr_tensor = tf.convert_to_tensor(Re[0]*Pr[0])


#%%

wdot_map_dic = sio.loadmat('../Data/wdot_map_mod.mat')
wdot_map = wdot_map_dic['wdot_map'].reshape(-1)
wdot_map = np.append(0, wdot_map)
wdot_map = np.append(wdot_map, 0)
# wdot_map = tf.convert_to_tensor(wdot_map)

theta_map_dic = sio.loadmat('../Data/theta_map_mod.mat')
theta_map = theta_map_dic['theta_map'].reshape(-1)
theta_map = np.append(0, theta_map)
theta_map = np.append(theta_map,1)
# theta_map = tf.convert_to_tensor(theta_map)


#%%


layers = [1, 30, 30, 30, 30, 30, 30, 2]        # For initializing the field
hid_activation = 'tanh'
out_activation = 'tanh'

model = PINN_PDE(hid_activation,
                 out_activation,
                 x_inlet, x_outlet,
                 theta_inlet, u_inlet,
                 theta_outlet, u_outlet,
                 x_f, 
                 X_data, 
                 ub, lb,
                 xmax, U_IN, 
                 layers,
                 [RHO_IN, alpha, lambda_by_CP],
                 [scale_max, scale_min],
                 scale_out,
                 theta_data_tensor,
                 U_data_tensor,
                 rho_data_tensor,
                 RePr_tensor,
                 wdot_map,
                 theta_map)

#%%

adam_iterations = 5000         # Number of training steps 
lbfgs_max_iterations = 10000   # Max iterations for lbfgs

# Below two variables are used for piecewise learning rate scheduler for ADAM
lr_list = [1e-3, 1e-4]    ## Learnig rate
lr_epochs = [200]        ## Epochs at which the learning rate should change


#%%

#### Training
checkpoint_str = './Checkpoint/1D_flame_theta_as_data_loss'

hist = model.train(adam_iterations, lbfgs_max_iterations, checkpoint_str, lr_epochs, lr_list)
# 


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
ax.semilogy(model.full_loss_list[1], linewidth=3, label='Mass cons loss', 
            color='#ff7f0e')
ax.semilogy(model.full_loss_list[2], linewidth=3, label='Progress loss',
            color='#8c564b') 
ax.semilogy(model.full_loss_list[3], linewidth=3, label='Dirichlet loss',
            color='#9467bd')
ax.semilogy(model.full_loss_list[4], linewidth=3, label='Data loss',
            color='#d62728')



ax.set_xlabel('Epochs')
ax.set_ylabel('MSE')

# ax.set_ylim([1e-6, 1e9])

ax.legend(ncol=2)
plt.tight_layout()
plt.show()


#%%   Loading trained weights (Will be useful when the models are trained on clusters)

model.model.load_weights(checkpoint_str)
# 
#%% Field prediction

theta_pred, u_pred = model.predict(X_colloc_scaled[:,0])

## Rescaling back from the non-dimensional form to the dimensional form
theta_pred = np.array(theta_pred)
u_pred = np.array(u_pred)*U_IN
rho_pred = RHO_IN/(1 + alpha*theta_pred/(1-alpha))     
T_pred = theta_pred*(T.max() - T.min()) + T.min()



#%%  Plotting the predictions

plt.rc('font', family='serif')
plt.rc('font', size=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig,ax = plt.subplots(1,1, figsize=(12,3.5))


ax1 = ax.twinx()

#### Density plots
ax.plot(z,rho, label='Cantera density', linewidth=3, color='#cc0000')
ax.plot(X_colloc[10:-10:10],rho_pred[10:-10:10], 'o', label='PINN density', 
        linewidth=3, color='#cc0000', markersize=10, fillstyle='none')
ax.set_xlabel('Length')
ax.set_ylabel(r'Density $(kg/m³)$')
ax.legend(ncol=1,handleheight=1.4, labelspacing=0.0, handletextpad=0.2,
                borderpad=0.2, loc='best')
ax.set_xticks([0, 0.004, 0.008, 0.012, 0.016, 0.02])


### Velocity plots
ax1.plot(z,u, label='Cantera velocity', linewidth=3, color='#0000CD')
ax1.plot(X_colloc[10:-10:10],u_pred[10:-10:10], 'd', label='PINN velocity', linewidth=3, color='#0000CD',
          markersize=10, fillstyle='none')
ax1.set_ylabel(r'Velocity $(m/s)$')
ax1.legend(ncol=1,handleheight=1.4, labelspacing=0.0, handletextpad=0.2,
                borderpad=0.2, loc='best')


### Temperature plots
# ax1.plot(z,T, label='Cantera temperature', linewidth=3, color='#228B22')
# ax1.plot(X_colloc[10:-10:10],T_pred[10:-10:10], 'd', label='PINN temperature', linewidth=3, color='#228B22',
#           markersize=10, fillstyle='none')
# ax1.set_ylabel(r'Temperature $(K)$')
# ax1.legend(ncol=1,handleheight=1.4, labelspacing=0.0, handletextpad=0.2,
#                 borderpad=0.2, loc='best')



plt.tight_layout()
plt.show()





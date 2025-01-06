# %% Imports
import numpy as np
import pickle
import tensorflow as tf
from scipy import io as sio
import time
from datetime import datetime
# from pyDOE import lhs
import matplotlib.pyplot as plt
import sys
from scipy.stats import qmc
from scipy.interpolate import griddata
import h5py
from matplotlib.ticker import FormatStrFormatter
from scipy import interpolate
from PDE_cyl import PINN_PDE
import glob, os, shutil, h5py


# %% Define Constants
Re = np.array([100])
gamma = 0.000482
u_in = 1
v_in = 0


# %% Read paraview (simulation) data

# solutDir = './SOLUT/'
# solutName = 'solut_'

# # get file names
# fileNames = glob.glob(solutDir+solutName+'*.h5')
# # sort fileNames
# timeStamps = [int(k.split('_')[-1].split('.')[0]) for k in fileNames]
# sortingIDX = np.argsort(timeStamps)
# sortedtimeStamps = [timeStamps[k] for k in sortingIDX]
# sortedFileNames = [fileNames[k] for k in sortingIDX]

# # The SOLUT folder consisits of solution at every time stamp
# # In the intial few time stamps, the simulation is transient and converges 
# # to a steady state solution after few time stamps. 
# # We can use the data from the last h5 file in sortedFileNames 
# # as this will correspond to the steady state

# for k,m in zip(sortedFileNames,range(0,len(sortedFileNames))):
#     dat = h5py.File(k,'r')
#     rho = dat['GaseousPhase']['rho'][:]
#     rhou = dat['GaseousPhase']['rhou'][:]
#     rhov = dat['GaseousPhase']['rhov'][:]
#     rhow = dat['GaseousPhase']['rhow'][:]
#     U_data = rhou/rho
#     V_data = rhov/rho
#     W_data = rhow/rho
#     P_data = dat['Additionals']['pressure'][:]
    
# # YOU CAN GET THE KEY NAMES BY THE FOLLOWING COMMAND:
# print(dat.keys())
# # OR
# print(dat['GaseousPhase'].keys())
# print(dat['Additionals'].keys())
# # and acceess them as above in the loop

# # you can get all the coordinates and the triangulation of the mesh from the mesh file
# # ATTENTION: for 2D, the plane is probably mirrored in z
# meshFile = './MESH/mesh.mesh.h5'
# mesh = h5py.File(meshFile,'r')
# x_data = mesh['Coordinates']['x'][:]
# y_data = mesh['Coordinates']['y'][:]
# z_data = mesh['Coordinates']['z'][:]

# # %% Save to CSV
# np.savetxt(
#     "solution.csv", 
#     csv_data, 
#     delimiter=",", 
#     header="x,y,z,u,v,p", 
#     comments=''
# )


# %% Load the CSV file

data = np.loadtxt("solution.csv", delimiter=",", skiprows=1)

# Extract columns (assuming columns in order: x,y,z,u,v,p)
x_data = data[:, 0]
y_data = data[:, 1]
z_data = data[:, 2]
U_data = data[:, 3]
V_data = data[:, 4]
P_data = data[:, 5]


# %% Bounds
# Set number of data points

# Set boundary, taken from min/max of available data
x_l = x_data.min()
x_L = x_data.max()

y_b = y_data.min()
y_B = y_data.max()

# positioning the cylinder's center, here center of the domain
x_c, y_c = 0, 0

# Lower bounds
lbx = np.array([x_l])   # Global Lower bound for time and x variables
lby = np.array([y_b])   # Global Lower bound for time and x variables

# Upper bounds
ubx = np.array([x_L])  # Global Upper bound for time and x variables
uby = np.array([y_B])  # Global Upper bound for time and x variables

lb = np.array([lbx, lby])
ub = np.array([ubx, uby])

# This is used for scaling from (0,1) to (-1,1)
scale_out = [2.0, 1.0]
scale_max = U_data.max()
scale_min = U_data.min()


# %% Domain Boundary Points

N_bound_x = 1000
N_bound_y = int(N_bound_x * (y_B-y_b/x_L-x_l))
N_c_border = int(N_bound_x/10)
N_c_internal = int(N_bound_x)

# Frame boundaries
x_border = np.linspace(x_l, x_L, N_bound_x).reshape(-1, 1)  # x points for left/right borders
y_border = np.linspace(y_b, y_B, N_bound_y).reshape(-1, 1)  # y points for bottom/top borders

# Bottom boundary (y = y_b)
y_bbound = np.hstack((x_border, np.full_like(x_border, y_b)))

# Top boundary (y = y_B)
y_Bbound = np.hstack((x_border, np.full_like(x_border, y_B)))

# Left boundary (x = x_l)
x_lbound = np.hstack((np.full_like(y_border, x_l), y_border))

# Right boundary (x = x_L)
x_Lbound = np.hstack((np.full_like(y_border, x_L), y_border))

# Cylinder surface points (approximate using polar coordinates)
phi_c = np.linspace(0, 2 * np.pi, N_c_border).reshape(-1, 1)
x_c_bound = gamma * np.cos(phi_c)
y_c_bound = gamma * np.sin(phi_c)
cylinder_points_boundary = np.hstack((x_c_bound, y_c_bound))

# Generate LHS points in the bounding box [-gamma, gamma] x [-gamma, gamma]
sampler = qmc.LatinHypercube(d=2)  # 2D LHS sampling
lhs_samples = sampler.random(N_c_internal)  # Generate N_points in [0, 1] x [0, 1]

# Rescale to the cylinder's bounding box [-gamma, gamma]
x_samples = 2 * gamma * lhs_samples[:, 0] - gamma  # Scale to [-gamma, gamma]
y_samples = 2 * gamma * lhs_samples[:, 1] - gamma  # Scale to [-gamma, gamma]

# Filter points to keep only those inside the cylinder
dist_squared = (x_samples - x_c)**2 + (y_samples - y_c)**2  # Distance to the center
mask = dist_squared < gamma**2  # Keep points within the circle
x_inside = x_samples[mask]
y_inside = y_samples[mask]
cylinder_points_inside = np.hstack((x_inside.reshape(-1, 1), y_inside.reshape(-1, 1)))


# %% Formulate Boundary Conditions

# Inlet boundary (x = x_l)
# u_inlet = np.squeeze(10 * np.sin(np.pi * (y_border - y_b) / (y_B - y_b)))  # Normalize y to [0, π] range
u_inlet = np.full(len(y_border), u_in)  # u velocity at the inlet
v_inlet = np.full(len(y_border), v_in)  # v velocity at the inlet

# No-slip condition on the bottom boundary (y = y_b)
u_b = np.zeros(len(x_border))  # u velocity at the bottom
v_b = np.zeros(len(x_border))  # v velocity at the bottom

# No-slip condition on the top boundary (y = y_B)
u_B = np.zeros(len(x_border))  # u velocity at the top
v_B = np.zeros(len(x_border))  # v velocity at the top

# %% Create Boundary Data Set

# Bottom boundary (y = y_b)
bottom_points = y_bbound  # (x, y) 
bottom_conditions = np.hstack((u_b.reshape(-1, 1), v_b.reshape(-1, 1)))  # (u, v)
bottom_data = np.hstack((bottom_points, bottom_conditions))  # (x, y, u, v)

# Top boundary (y = y_B)
top_points = y_Bbound
top_conditions = np.hstack((u_B.reshape(-1, 1), v_B.reshape(-1, 1)))
top_data = np.hstack((top_points, top_conditions))

# Inlet boundary (x = x_l)
inlet_points = x_lbound
inlet_conditions = np.hstack((u_inlet.reshape(-1, 1), v_inlet.reshape(-1, 1)))
inlet_data = np.hstack((inlet_points, inlet_conditions))

# Combine cylinder boundary and internal points
cylinder_points = np.vstack((cylinder_points_boundary, cylinder_points_inside))
cylinder_conditions = np.vstack((np.zeros_like(cylinder_points[:, 0]), np.zeros_like(cylinder_points[:, 1]))).T
cylinder_data = np.vstack((cylinder_points, cylinder_conditions))

## Transform into PINN class object
# Combine all boundary data
# boundary_data = np.vstack((bottom_data, top_data, inlet_data, cylinder_data))

# Bottom boundary (y = y_b)
x_boundary_bottom = y_bbound[:, 0]  
y_boundary_bottom = y_bbound[:, 1]  

# Top boundary (y = y_B)
x_boundary_top = y_Bbound[:, 0]  
y_boundary_top = y_Bbound[:, 1]  

# Inlet boundary (x = x_l)
x_boundary_inlet = x_lbound[:, 0]
y_boundary_inlet = x_lbound[:, 1]

# Cylinder boundary
x_cylinder = cylinder_points[:, 0]
y_cylinder = cylinder_points[:, 1]  
u_cylinder = cylinder_conditions[:, 0]
v_cylinder = cylinder_conditions[:, 1]

# Combine boundary data
x_boundary = np.concatenate([x_boundary_bottom, x_boundary_top, x_boundary_inlet])
y_boundary = np.concatenate([y_boundary_bottom, y_boundary_top, y_boundary_inlet])
u_boundary = np.concatenate([u_b, u_B, u_inlet])
v_boundary = np.concatenate([v_b, v_B, v_inlet])


# %% Plot Cylinder points

# plt.figure(figsize=(6, 6))
# plt.scatter(x_inside, y_inside, s=5, c='red', label='LHS points inside cylinder')
# circle = plt.Circle((x_c, y_c), gamma, color='blue', fill=False, linewidth=2, label='Cylinder boundary')
# plt.gca().add_artist(circle)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title('LHS Points Inside the Cylinder')
# plt.legend()
# plt.show()

# print(f"Number of points inside the cylinder: {len(x_inside)} (target was {N_c_internal})")

# %% Incoorperate experimental Data points

experimental_data = np.hstack((
    x_data.reshape(-1, 1),
    y_data.reshape(-1, 1),
    U_data.reshape(-1, 1),
    V_data.reshape(-1, 1)
))

x_data = experimental_data[:, 0]  # x-coordinates
y_data = experimental_data[:, 1]  # y-coordinates
u_data = experimental_data[:, 2]  # u velocities
v_data = experimental_data[:, 3]  # v velocities

# %% Combine all data

# combined_data = np.vstack((boundary_data, experimental_data))  # Combine all rows


# %% Collocation points

# Sampled collocation points based on LHS
lhs = qmc.LatinHypercube(d=2, seed=1234)
N_colloc = 5000

Colloc_train = np.array(lhs.random(N_colloc))
Colloc_train[:, 0] = x_l + (x_L - x_l) * Colloc_train[:, 0]
Colloc_train[:, 1] = y_b + (y_B - y_b) * Colloc_train[:, 1]
# Colloc_train1_scaled = (Colloc_train1)/[xmax]

# exclude the cylinder
Colloc_train = Colloc_train[np.sum((Colloc_train - np.array([x_c, y_c]))**2, axis=1) >= gamma**2]

X_colloc_scaled = Colloc_train/np.array([x_L, y_B])

# Extract collocation points
x_collocation = Colloc_train[:, 0]
y_collocation = Colloc_train[:, 1]


# %% Plot domain

# # Create the plot
# plt.figure(figsize=(10, 10))  # Set figure size

# # Plot each type of point using scatter with a unique color and label
# plt.scatter(x_data, y_data, color='red', s=10, label='Data Points', alpha=0.6)         # Data points in red
# plt.scatter(x_cylinder, y_cylinder, color='blue', s=10, label='Cylinder Points')       # Cylinder boundary points in blue
# plt.scatter(x_collocation, y_collocation, color='black', s=10, label='Collocation Points', alpha=1)  # Collocation points in green

# # Add labels, title, and legend
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
# plt.title('Distribution of Data, Cylinder, and Collocation Points in the Domain')
# plt.legend(loc='best')  # Show legend at best location
# plt.axis('equal')  # Set equal axis for correct aspect ratio
# plt.grid(True, linestyle='--', alpha=0.5)  # Optional: Add a grid for better visualization

# # Show the plot
# plt.show()

# %% Plot velocities of initial conditions

# x = x_boundary
# y = y_boundary
# u = u_boundary
# v = v_boundary

# # x = x_cylinder
# # y = y_cylinder
# # u = u_cylinder
# # v = v_cylinder

# # x = x_data
# # y = y_data
# # u = u_data
# # v = v_data

# # Plot for u
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, c=u, cmap='coolwarm', s=50, alpha=0.8)  # Color-coded by u
# plt.colorbar(label='Velocity u')
# plt.xlabel('x position')
# plt.ylabel('y position')
# plt.title('Spatial distribution of velocity u')
# plt.axis('equal')  # Set equal axis for correct aspect ratio
# plt.show()

# # Plot for v
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, c=v, cmap='coolwarm', s=50, alpha=0.8)  # Color-coded by v
# plt.colorbar(label='Velocity v')
# plt.xlabel('x position')
# plt.ylabel('y position')
# plt.title('Spatial distribution of velocity v')
# plt.axis('equal')  # Set equal axis for correct aspect ratio
# plt.show()


# %% Coversion to tensorflow

# Convert boundary points and conditions to tensors
x_boundary_tensor = tf.convert_to_tensor(x_boundary)
y_boundary_tensor = tf.convert_to_tensor(y_boundary)
u_bc_tensor = tf.convert_to_tensor(u_boundary)
v_bc_tensor = tf.convert_to_tensor(v_boundary)

# Convert cylinder boundary points to tensors
x_cylinder_tensor = tf.convert_to_tensor(x_cylinder)
y_cylinder_tensor = tf.convert_to_tensor(y_cylinder)
u_cylinder_tensor = tf.convert_to_tensor(u_cylinder)
v_cylinder_tensor = tf.convert_to_tensor(v_cylinder)

# Convert experimental data to tensors
x_data_tensor = tf.convert_to_tensor(x_data)
y_data_tensor = tf.convert_to_tensor(y_data)
u_data_tensor = tf.convert_to_tensor(u_data)
v_data_tensor = tf.convert_to_tensor(v_data)

# Convert collocation points to tensors
x_collocation_tensor = tf.convert_to_tensor(x_collocation)
y_collocation_tensor = tf.convert_to_tensor(y_collocation)

# Constants
Re_tensor = tf.convert_to_tensor(Re)
gamma_tensor = tf.convert_to_tensor(gamma)

# Scaling
scale_max_tensor = tf.convert_to_tensor(scale_max)
scale_min_tensor = tf.convert_to_tensor(scale_min)

ub_tensor = tf.convert_to_tensor(ub)
lb_tensor = tf.convert_to_tensor(lb)

# %% Create PINN class object

layers = [2, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 2]        # For initializing the field
hid_activation = 'tanh'
out_activation = 'tanh'

model = PINN_PDE(
    hid_activation=hid_activation,
    out_activation=out_activation,
    x_boundary=x_boundary_tensor,
    y_boundary=y_boundary_tensor,
    u_bc=u_bc_tensor,
    v_bc=v_bc_tensor,
    x_cylinder=x_cylinder_tensor,
    y_cylinder=y_cylinder_tensor,
    u_cylinder=u_cylinder_tensor,
    v_cylinder=v_cylinder_tensor,
    gamma=gamma_tensor,
    x_f=x_collocation_tensor,  # --> lhs
    y_f=y_collocation_tensor,  # --> lhs
    x_data=x_data_tensor,  # 1 2 1 transfer
    y_data=y_data_tensor,
    u_data=u_data_tensor,
    v_data=v_data_tensor,
    ub=ub_tensor,
    lb=lb_tensor,
    Re=Re_tensor,
    scale=(scale_max_tensor, scale_min_tensor),
    scale_out=scale_out,
    layers=layers
)


# %% Training
now = datetime.now()
start_timestamp = now.strftime("%y-%m-%d_%H%M")

adam_iterations = 5000
lbfgs_max_iterations = 50000

"""
Time table:
100 + 500 = 3m
500 + 2500 = 16m
1000 + 10000 = ~2.5h
5000 + 50000 = 6h 
5000 + 50000 (inkl. yS) = 11.5h

"""

# Below two variables are used for piecewise learning rate scheduler for ADAM
lr_list = [1e-3, 1e-4, 1e-5]    ## Learnig rate
lr_epochs = [100,1000]        ## Epochs at which the learning rate should change

checkpoint_str = f'./Checkpoint/cylinder_flow_checkpoints_{start_timestamp}'

hist = model.train(adam_iterations, lbfgs_max_iterations, checkpoint_str, lr_epochs, lr_list)

now = datetime.now()
end_timestamp = now.strftime("%y-%m-%d_%H%M")

print(f"Training start time: {start_timestamp} \n Training end time: {end_timestamp}. \n")

# %% Save losses to lists
now = datetime.now()
timestamp = now.strftime("%y-%m-%d_%H%M")

# Directory for saving the loss data
output_dir = "./loss_data"
os.makedirs(output_dir, exist_ok=True)

# File path for saving the losses
loss_file = os.path.join(output_dir, f"losses_{timestamp}.pkl")

# Save the list of losses to a pickle file
with open(loss_file, 'wb') as f:
    pickle.dump(model.full_loss_list, f)

print(f"Loss data saved to {loss_file}")

# %% load loss lists

# # Load the .pkl file
# pkl_file = './loss_data/losses_24-12-15_1439.pkl'  # Replace with your file path

# with open(pkl_file, 'rb') as file:
#     saved_data = pickle.load(file)

# # Extract the loss list from the loaded data
# full_loss_list = saved_data

# %% Loss curve of loaded losses
# now = datetime.now()
# timestamp = now.strftime("%y-%m-%d_%H%M")

# output_dir = "./plots"
# os.makedirs(output_dir, exist_ok=True)

# plt.rc('font', family='serif')
# plt.rc('font', size=14)
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)

# ##### Plotting losses

# fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))

# # Plot losses from loaded data
# ax.semilogy(full_loss_list[0], linewidth=3, label='Total loss', color='#1f77b4')  # Blue
# ax.semilogy(full_loss_list[1], linewidth=3, label='Mass cons loss', color='#ff7f0e')  # Orange
# ax.semilogy(full_loss_list[2], linewidth=3, label='Momentum cons loss (x)', color='#2ca02c')  # Green
# ax.semilogy(full_loss_list[3], linewidth=3, label='Momentum cons loss (y)', color='#9467bd')  # Purple
# ax.semilogy(full_loss_list[4], linewidth=3, label='Boundary loss', color='#d62728')  # Red
# ax.semilogy(full_loss_list[5], linewidth=3, label='Cylinder loss', color='#17becf')  # Cyan
# ax.semilogy(full_loss_list[6], linewidth=3, label='Data loss', color='#bcbd22')  # Yellow-green

# # Add vertical line at epoch = 200
# ax.axvline(x=200, color='gray', linestyle='--', linewidth=2, label='Epoch 200')

# ax.set_xlabel('Epochs')
# ax.set_ylabel('MSE')

# ax.legend(ncol=2)
# plt.tight_layout()

# # Save the plot
# output_file = os.path.join(output_dir, f"loss_curve_{timestamp}.png")
# plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save with high resolution

# plt.show()

# %% Loss curve
# now = datetime.now()
# timestamp = now.strftime("%y-%m-%d_%H%M")

# output_dir = "./plots"
# os.makedirs(output_dir, exist_ok=True)

# plt.rc('font', family='serif')
# plt.rc('font', size=14)
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)


# ##### Plotting losses
# fig, ax = plt.subplots(1,1, figsize=(9,4.5))



# ax.semilogy(model.full_loss_list[0], linewidth=3, label='Total loss',
#             color='#1f77b4')  # Blue
# ax.semilogy(model.full_loss_list[1], linewidth=3, label='Mass cons loss', 
#             color='#ff7f0e')  # Orange
# # ax.semilogy(model.full_loss_list[2], linewidth=3, label='Momentum cons loss (x)',
# #             color='#2ca02c')  # Green
# # ax.semilogy(model.full_loss_list[3], linewidth=3, label='Momentum cons loss (y)',
# #             color='#9467bd')  # Purple
# ax.semilogy(model.full_loss_list[2], linewidth=3, label='Boundary loss',
#             color='#d62728')  # Red
# ax.semilogy(model.full_loss_list[3], linewidth=3, label='Cylinder loss',
#             color='#17becf')  # Cyan
# ax.semilogy(model.full_loss_list[4], linewidth=3, label='Data loss',
#             color='#000000')  # Yellow-green

# # Add vertical line at epoch = 200
# ax.axvline(x=lr_epochs[0], color='gray', linestyle='--', linewidth=2, label='Epoch 200')

# ax.set_xlabel('Epochs')
# ax.set_ylabel('MSE')

# # ax.set_ylim([1e-6, 1e9])

# ax.legend(ncol=2)
# plt.tight_layout()

# # Save the plot
# output_file = os.path.join(output_dir, f"loss_curve_{timestamp}.png")
# plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save with high resolution

# plt.show()


# %% Loading trained weights (Will be useful when the models are trained on clusters)
# checkpoint_str = f'./Checkpoint/'
# model.model.load_weights(checkpoint_str)
# 
# %% Field prediction

# u_pred, v_pred = model.predict(x_collocation_tensor, y_collocation_tensor)

## Rescaling back from the non-dimensional form to the dimensional form
# u_pred = np.array(u_pred)
# v_pred = np.array(v_pred)

# %% Plotting the u-v field

# pred_dir = "./plots/predictions"
# os.makedirs(pred_dir, exist_ok=True)

# plt.figure(figsize=(10, 10))  # Set figure size

# # Predict at experimental data points
# x_data = x_data_tensor.numpy()
# y_data = y_data_tensor.numpy()
# xy_data = np.hstack((x_data, y_data))  # Combine x and y
# u_data_pred, v_data_pred = model.predict(x_data, y_data)

# # Predict at collocation points
# x_f = x_collocation_tensor.numpy()
# y_f = y_collocation_tensor.numpy()
# xy_colloc = np.hstack((x_f, y_f))
# u_colloc_pred, v_colloc_pred = model.predict(x_f, y_f)

# # Create a uniform grid for x and y (100x100 grid as an example)
# x_grid = np.linspace(np.min(x_data), np.max(x_data), 100)
# y_grid = np.linspace(np.min(y_data), np.max(y_data), 100)
# X, Y = np.meshgrid(x_grid, y_grid)

# # Compute velocity magnitude at each data point
# velocity_magnitude = np.sqrt(u_colloc_pred**2 + v_colloc_pred**2)

# # Interpolate the scattered velocity magnitude data onto the uniform grid
# Z = griddata((x_f, y_f), velocity_magnitude, (X, Y), method='cubic')

# # Plot contour of velocity magnitude
# plt.contourf(X, Y, Z, levels=50, cmap='viridis')
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'u/v-pred: yB+yC+yD+mass, adam: {adam_iterations}, LBFGS: {lbfgs_max_iterations}')
# plt.axis('equal')  # Set equal axis for correct aspect ratio

# # Save the plot
# output_file = os.path.join(pred_dir, f"prediction_{timestamp}.png")
# plt.savefig(output_file, dpi=300, bbox_inches='tight')

# plt.show()


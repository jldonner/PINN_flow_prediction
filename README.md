# Thermoacoustic Project — PINN Cylinder Flow

This project implements a **Physics-Informed Neural Network (PINN)** to model 2-D incompressible flow around a cylinder. The PINN combines physics laws (continuity + momentum), boundary conditions (walls, inlet, cylinder), and velocity data (from HDF5 simulation snapshots) into a single loss function.

---

## Features
- Solves steady incompressible Navier–Stokes in 2D (u, v fields).  
- Enforces:  
  - Mass conservation  
  - Momentum conservation (x and y)  
  - Boundary conditions (inlet, top/bottom walls, cylinder no-slip)  
  - Data matching (velocity fields from HDF5 files)  
- Training with **ADAM** (with learning rate schedule) + **L-BFGS** optimization.  
- Generates loss curves, checkpoints, and velocity magnitude plots.

---

## Repository Structure
```
.
├── main.py               # Run script: data loading → training → plots
├── PDE_cyl.py            # PINN model: physics & boundary losses
├── optimizer/
│   └── NeuralNet_optimizer.py   # Base classes for PINNs
├── MESH/                 # Input mesh (mesh.mesh.h5)
├── SOLUT/                # Solution snapshots (solut_*.h5)
├── plots/                # Saved figures
├── loss_data/            # Pickled loss history
└── Checkpoint/           # Model checkpoints
```

---

## Installation
```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Example `requirements.txt`:
```
tensorflow>=2.13,<2.16
numpy
scipy
matplotlib
h5py
```

---

## Usage
1. Place your input files:
   - `MESH/mesh.mesh.h5` (coordinates: x, y, z)  
   - `SOLUT/solut_*.h5` (solution snapshots with velocity data)  
2. Adjust constants in `main.py` if needed:
   ```python
   Re = np.array([100])
   gamma = 0.000482   # cylinder radius
   ```
3. Run:
   ```bash
   python main.py
   ```

---

## Outputs
- Model checkpoints: `Checkpoint/cylinder_flow_checkpoints_<timestamp>.weights.h5`  
- Loss history: `loss_data/losses_<timestamp>.pkl`  
- Training curves: `plots/loss_curve_<timestamp>.png`  
- Prediction plots: `plots/predictions/prediction_<timestamp>.png`  

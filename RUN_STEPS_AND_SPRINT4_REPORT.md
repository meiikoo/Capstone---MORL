# CARLA MO Project - Run Steps and Weekly Report

## 0) Prerequisites

- Open **Command Prompt**.
- Go to the project folder:

```bat
cd C:\Users\12482\OneDrive\Documents\CARLA_0.9.16
```

- Make sure CARLA server executable exists (`CarlaUE4.exe`) and Python can run.

---

## 1) Run `carla_MO.py` (MO environment smoke test)

### Terminal A - start CARLA server

```bat
cd C:\Users\12482\OneDrive\Documents\CARLA_0.9.16
.\CarlaUE4.exe
```

### Terminal B - run MO script

```bat
cd C:\Users\12482\OneDrive\Documents\CARLA_0.9.16
python carla_MO.py
```

What it does:
- Connects to CARLA, spawns ego vehicle + LiDAR + collision sensor.
- Runs test steps and prints vector reward:
  - Efficiency
  - Safety
  - Stability
- Prints debug metrics (speed, lateral acceleration, rollover risk, min LiDAR distance).

---

## 2) Run `carla_mo_gym_env.py` (Gym adapter sanity check)

`carla_mo_gym_env.py` is mainly an adapter module (normally imported by training scripts), so use this command for a quick check.

### Terminal A - start CARLA server

```bat
cd C:\Users\12482\OneDrive\Documents\CARLA_0.9.16
.\CarlaUE4.exe
```

### Terminal B - one-line adapter test

```bat
cd C:\Users\12482\OneDrive\Documents\CARLA_0.9.16
python -c "from carla_mo_gym_env import CarlaMOGymEnv; env=CarlaMOGymEnv(town='Town04', episode_length=20); obs,info=env.reset(); print('obs_shape=', obs.shape, 'reward_dim=', env.reward_space.shape[0]); env.close()"
```

Expected:
- It prints observation shape and reward dimension.
- No connection/runtime error means adapter is working.

---

## 3) Run `train_pcn_carla.py` (PCN baseline experiment)

### First-time dependency setup (if needed)

```bat
python -m pip install wandb pymoo mo-gymnasium gymnasium
```

### Terminal A - start CARLA server

```bat
cd C:\Users\12482\OneDrive\Documents\CARLA_0.9.16
.\CarlaUE4.exe
```

### Terminal B - run PCN training

```bat
cd C:\Users\12482\OneDrive\Documents\CARLA_0.9.16
python train_pcn_carla.py --morl_repo "C:\Users\12482\Documents\GitHub\morl-baselines" --total_timesteps 2000
```

Optional quick test:

```bat
python train_pcn_carla.py --morl_repo "C:\Users\12482\Documents\GitHub\morl-baselines" --total_timesteps 500
```

What it does:
- Imports PCN from `morl-baselines`.
- Uses `CarlaMOGymEnv` to provide Gym/MO-style API.
- Trains PCN on 3-objective CARLA reward vector.

---

## Weekly Report

This week we completed the first MORL integration pipeline for CARLA. We finalized `carla_MO.py` so it can consistently spawn the vehicle, run steps, and output a 3D reward vector for efficiency, safety, and stability with relevant diagnostics. Then we built `carla_mo_gym_env.py` to wrap our CARLA environment into a Gymnasium-compatible interface required by MORL-Baselines. Finally, we created `train_pcn_carla.py` to run the PCN baseline directly on our CARLA setup, verified dependencies, fixed a NumPy 2.x compatibility issue in the external PCN implementation, and successfully executed a PCN training run end-to-end.


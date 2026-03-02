import gymnasium
import numpy as np
import highway_env
from gymnasium import spaces
import math

class MOHighwayWrapper(gymnasium.Wrapper):
    """
    Wraps the highway-env to return a simple multi-objective reward vector:
    [efficiency, safety, stability]
    
    Updated to use Continuous Actions and Static Stability Factor (SSF)
    """
    
    # Vehicle parameters for SSF calculation
    TRACK_WIDTH = 1.8  # meters (T)
    CG_HEIGHT = 0.5    # meters (h)
    GRAVITY = 9.81     # m/s^2 (g)
    
    # SSF Threshold = T / 2h
    # If Lateral Acceleration > SSF * g, rollover risk is high
    SSF_LIMIT_G = (TRACK_WIDTH / (2 * CG_HEIGHT)) * GRAVITY

    #Lidar Safety Thresholds
    CRITICAL_DISTANCE = 5.0  # meters (critical distance for safety)
    WARNING_DISTANCE = 15.0  # meters (warning distance for safety)
    SAFE_DISTANCE = 30.0     # meters (safe distance for safety)

    def __init__(self, env):
        super().__init__(env)
        # Define the reward space: 3 objectives
        # shape=(3,) means we return 3 numbers
        self.reward_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.unwrapped.reward_space = self.reward_space
        
        # State tracking for derivatives
        self.last_heading = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset state tracking
        self.last_heading = None
        return obs, info
        
    def _calculate_rollover_risk(self, info):
        """
        Calculates the risk of rollover based on Lateral Acceleration.
        Returns a value between 0.0 (Safe) and 1.0+ (Risk/Rollover).
        """
        # Get current vehicle state from the unwrapped environment
        vehicle = self.unwrapped.vehicle
        
        current_heading = vehicle.heading
        current_speed = vehicle.speed
        
        # Calculate Yaw Rate (d_heading / dt)
        # Simulation step is usually 1/15 seconds
        dt = 1.0 / self.unwrapped.config["simulation_frequency"]
        
        if self.last_heading is not None:
            # Handle angle wrapping (-pi to pi)
            diff = current_heading - self.last_heading
            # Normalize angle difference to [-pi, pi]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            yaw_rate = diff / dt
        else:
            yaw_rate = 0.0
            
        # Update last heading for next step
        self.last_heading = current_heading
        
        # Calculate Lateral Acceleration (a_y = v * yaw_rate)
        # This is the centripetal force in m/s^2
        lat_accel = current_speed * yaw_rate
        
        # Calculate Risk Factor
        # Risk = |Lat_Accel| / (SSF * g)
        risk = abs(lat_accel) / self.SSF_LIMIT_G
        
        return risk, lat_accel
    
    def calculate_lidar_safety(self, lidar_obs):
        max_range = 50.0  # meters
        distances = lidar_obs * max_range

        min_distance = np.min(distances)

        #risk increases non-linearly as we get closer to the obstacles
        if min_distance > self.SAFE_DISTANCE:
            collision_risk = 0.0
        elif min_distance > self.WARNING_DISTANCE:
            #warning zone
            collision_risk = 0.3 * (1 - (min_distance - self.WARNING_DISTANCE) / 
                                   (self.SAFE_DISTANCE - self.WARNING_DISTANCE))
        elif min_distance > self.CRITICAL_DISTANCE:
            #critical zone
            collision_risk = 0.3 +  0.4 * (1 - (min_distance - self.CRITICAL_DISTANCE) /
                                      (self.WARNING_DISTANCE - self.CRITICAL_DISTANCE))
        else:
            #collision imminent
            collision_risk = 0.7 + 0.3 * (1 - min_distance / self.CRITICAL_DISTANCE)

        return collision_risk, min_distance


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        #extract observations (obs is a tuple):
        kinematics_obs = obs[0]
        lidar_obs = obs[1]

        # --- 1. Efficiency Reward (Speed) ---
        # info['speed'] in highway-env is usually the raw speed in m/s.
        # We manually normalize it here for the reward calculation.
        current_speed = info['speed']  

        # typical range from 0 to 35 m/s in highway-env
        min_speed = 0.0
        max_speed = 35 # average highway-env speed

        # Normalize 0.0 to 1.0
        efficiency_reward = np.clip((current_speed - min_speed) / (max_speed - min_speed), 0.0, 1.0)

        # --- 2. Safety Reward (Collision) with LIDAR---
        collision_risk, min_distance = self._calculate_lidar_safety(lidar_obs)

        if info['crashed']:
            safety_reward = -1.0
        else:
            safety_reward = 1.0 - collision_risk  # Higher risk reduces safety reward
        
        # --- 3. Stability Reward (SSF / Rollover Risk) ---
        risk, lat_accel = self._calculate_rollover_risk(info)
        
        # We penalize high risk.
        # If risk is 0 (straight line), reward is 0.0 (Neutral/Good)
        # If risk is 1.0 (tipping point), reward is -1.0 (Bad)
        # If risk > 1.0 (rollover), reward is < -1.0 (Very Bad)
        stability_reward = -1.0 * risk
        
        # Clip stability reward to -1.0 to keep it bounded like Efficiency (0 to 1) and Safety (-1 to 1).
        stability_reward = np.clip(stability_reward, -1.0, 0.0)

        # Combine into a vector [Efficiency, Safety, Stability]
        vec_reward = np.array([efficiency_reward, safety_reward, stability_reward], dtype=np.float32)
        
        # Add extra info for debugging
        info['lateral_accel'] = lat_accel
        info['rollover_risk'] = risk
        
        return obs, vec_reward, terminated, truncated, info

# --- Configuration for the Environment ---
config = {
    "observation": {
        "type": "TupleObservation",
        "observation_configs": [
        {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
            "order": "sorted"
        },
        {
            "type":"LidarObservation",
            "cells": 64,
            "maximum_range": 50,
            "normalise": True,
        }
        ]
    },
    "action": {
        # CHANGED: ContinuousAction for smooth steering
        "type": "ContinuousAction",
        # Limit steering to +/- 0.2 radians (~11 degrees) for highway safety
        "steering_range": [-0.2, 0.2],
        "longitudinal": True,
        "lateral": True,
    },
    "simulation_frequency": 15,  # [Hz] Needed for dt calc
    "policy_frequency": 1,       # [Hz]
    "duration": 40,              # Episode length in seconds
    "vehicles_density": 1.5,     # Traffic density
}

if __name__ == "__main__":
    # Test the environment
    env = gymnasium.make("highway-fast-v0", render_mode="rgb_array", config=config)
    mo_env = MOHighwayWrapper(env)
    
    print("Environment created with Continuous Actions and SSF Stability!")
    print(f"SSF Threshold (g): {mo_env.SSF_LIMIT_G:.2f} m/s^2")
    
    obs, info = mo_env.reset()

    kinematics_obs = obs[0]  # Kinematics part of the observation
    lidar_obs = obs[1]       # Lidar part of the observation
    
    print("\nRunning test steps...")
    for i in range(10):
        # Sample random continuous action
        # Action is now [acceleration, steering]
        action = mo_env.action_space.sample()
        
        obs, vec_reward, done, truncated, info = mo_env.step(action)
        
        kinematics_obs = obs[0]  # Kinematics part of the observation
        lidar_obs = obs[1]       # Lidar part of the observation
        lat_accel = info['lateral_accel']
        risk = info['rollover_risk']
        
        print(f"Step {i+1}: Action={action}, Reward={vec_reward}")
        print(f"  -> Lat Accel: {lat_accel:.2f}, Risk: {risk:.2f}")
        
    mo_env.close()

import gymnasium
import numpy as np
import highway_env
from gymnasium import spaces

class MOHighwayWrapper(gymnasium.Wrapper):
    """
    Wraps the highway-env to return a simple multi-objective reward vector:
    [efficiency, safety, stability]
    """
    def __init__(self, env):
        super().__init__(env)
        # Define the reward space: 3 objectives
        # shape=(3,) means we return 3 numbers
        self.reward_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.unwrapped.reward_space = self.reward_space
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- 1. Efficiency Reward (Speed) ---
        # info['speed'] is normalized speed (0 to 1).
        # We simply return the speed itself.
        # Fast = 1.0, Stopped = 0.0
        current_speed = info['speed']  

        #typical range from 0 to 35 in highway-env
        min_speed = 0.0
        max_speed = 35 # average highway-env speed

        efficiency_reward = np.clip((current_speed - min_speed) / (max_speed - min_speed), 0.0, 1.0)

        # --- 2. Safety Reward (Collision) ---
        # Simple logic:
        # If crashed: -1
        # If not crashed: +1 (reward for staying alive)
        if info['crashed']:
            safety_reward = -1.0
        else:
            safety_reward = 1.0
        
        # --- 3. Stability Reward (Lane Changes) ---
        # Simple logic:
        # If action is changing lanes (0 or 1): -1 penalty
        # Otherwise (driving straight): 0 (no penalty)
        # Actions in highway-env: 0: LCS, 1: LCR, 2: IDLE, 3: FASTER, 4: SLOWER
        if action == 0 or action == 1:
            stability_reward = -1.0
        else:
            stability_reward = 0.0
        
        # Combine into a vector [Efficiency, Safety, Stability]
        vec_reward = np.array([efficiency_reward, safety_reward, stability_reward], dtype=np.float32)
        
        # Return the vector reward instead of the original scalar reward
        return obs, vec_reward, terminated, truncated, info

# --- Configuration for the Environment ---
# keep the standard "highway-fast-v0" configuration but ensure we have the right observation type.
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": False,
        "order": "sorted"
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "duration": 40,              # Episode length in seconds
    "vehicles_density": 1.5,     # Traffic density (simpler than 2)
    "collision_reward": -1,        # Penalty for collisions (we handle this in our wrapper)
}

if __name__ == "__main__":
    # Test the environment to make sure our logic works
    env = gymnasium.make("highway-fast-v0", render_mode="rgb_array", config=config)
    mo_env = MOHighwayWrapper(env)
    
    print("Environment created!")
    obs, info = mo_env.reset()
    
    # Run 10 steps
    for i in range(10):
        # Pick a random action
        action = mo_env.action_space.sample()
        
        # Take a step
        obs, vec_reward, done, truncated, info = mo_env.step(action)
        
        # Print what happened
        action_name = ["LCS", "LCR", "IDLE", "FASTER", "SLOWER"][action]
        print(f"Step {i+1}: Action={action_name}, Reward={vec_reward}")
        # Expect:
        # Efficiency = Speed (0.0 to 1.0)
        # Safety = 1.0 (unless crashed)
        # Stability = -1.0 if LCS/LCR, else 0.0
        
    mo_env.close()

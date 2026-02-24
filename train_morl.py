import gymnasium
import mo_gymnasium as mo_gym
import numpy as np
import os

# Disable wandb logging for this simple example
os.environ["WANDB_MODE"] = "disabled"

from morl_baselines.multi_policy.pcn.pcn import PCN
from gymnasium.wrappers import FlattenObservation
from mo_multi_goal import MOHighwayWrapper, config

def train():
    # 1. Create the base environment
    # "highway-fast-v0" is the standard highway environment
    env = gymnasium.make("highway-fast-v0", render_mode="rgb_array", config=config)
    env = FlattenObservation(env)
    
    # 2. Wrap it with our simple Multi-Objective Wrapper
    env = MOHighwayWrapper(env)
    
    # Create an evaluation environment (same config)
    eval_env = gymnasium.make("highway-fast-v0", render_mode="rgb_array", config=config)
    eval_env = FlattenObservation(eval_env)
    eval_env = MOHighwayWrapper(eval_env)
    
    # 3. Create the MORL agent (PCN)
    # PCN learns a policy that can satisfy different trade-offs.
    print("Initializing PCN Agent...")
    
    agent = PCN(
        env=env,
        scaling_factor=np.array([1, 1, 1, 1]), # Simple scaling (1:1)
        learning_rate=1e-3,
        batch_size=32,
        project_name="highway-morl"
    )
    
    # 4. Train the agent
    print("Starting training...")
    agent.train(
        total_timesteps=10000, # Train for 10,000 steps (quick test)
        eval_env=eval_env,
        ref_point=np.array([0, -1, -1]), # Worst case reference point: [Stop, Crash, Change Lanes]
        known_pareto_front=None
    )
    
    print("Training finished!")
    
    # 5. Save the trained model
    agent.save("highway_pcn_model")

if __name__ == "__main__":
    train()

import gymnasium
import mo_gymnasium as mo_gym
import numpy as np
from morl_baselines.multi_policy.pcn.pcn import PCN
from highway_mo_env import MOHighwayWrapper, config

def train():
    # 1. Create the base environment
    # "highway-fast-v0" is the standard highway environment
    env = gymnasium.make("highway-fast-v0", render_mode="rgb_array", config=config)
    
    # 2. Wrap it with our simple Multi-Objective Wrapper
    env = MOHighwayWrapper(env)
    
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
        ref_point=np.array([0, -1, -1]), # Worst case reference point: [Stop, Crash, Change Lanes]
        known_pareto_front=None
    )
    
    print("Training finished!")
    
    # 5. Save the trained model
    agent.save("highway_pcn_model")

if __name__ == "__main__":
    train()

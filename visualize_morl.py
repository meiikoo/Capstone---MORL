import gymnasium
import mo_gymnasium as mo_gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
from morl_baselines.multi_policy.pcn.pcn import PCN
from highway_mo_env import MOHighwayWrapper, config

def visualize():
    # 1. Create the environment with "human" render mode (to see the window)
    # Use the exact same config as training
    env = gymnasium.make("highway-fast-v0", render_mode="human", config=config)
    env = FlattenObservation(env)
    
    # 2. Wrap it with our simple Multi-Objective Wrapper
    env = MOHighwayWrapper(env)
    
    # 3. Create the PCN agent 
    # Initialize with same parameters
    agent = PCN(
        env=env,
        scaling_factor=np.array([1, 1, 1, 1]),
        learning_rate=1e-3,
        batch_size=32
    )
    
    # 4. Load the trained weights
    print("Loading model...")
    # The file "highway_pcn_model" must exist
    agent.load("weights/highway_pcn_model.pt")
    print("Model loaded!")
    
    # 5. Define a Preference (Weighting)
    # [Efficiency, Safety, Stability]
    # Try changing:
    # [1.0, 0.0, 0.0] -> Crazy Speed
    # [0.0, 1.0, 0.0] -> Super Safe
    # [0.5, 0.5, 0.0] -> Balanced Mode
    desired_preference = np.array([1.0, 0.0, 0.0]) 
    print(f"Testing with Preference: {desired_preference}")

    # 6. Run 5 episodes
    for episode in range(5):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = np.zeros(3)
        
        while not (done or truncated):
            # Ask the agent: "Given this road (obs) and my preference, what should I do?"
            # eval() returns the best action
            action = agent.eval(obs, desired_preference)
            
            # Take the step
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Show the window
            env.render()
            
        print(f"Episode {episode+1} Finished. Total Reward Vector: {total_reward}")

    env.close()

if __name__ == "__main__":
    visualize()

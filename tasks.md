# MORL Environment Setup & Experimentation Tasks

## Highway-env Setup
Study how highway-env works and formulate an MO version of it. 

1. You can follow the "Getting Started" guide: https://highway-env.farama.org/quickstart/ 
   *There's an example using SB3’s DQN to train an agent. They say it took 25 min on their laptop to train the DQN. (What computer setup do you have available?)*

2. Take a look at the ways we can personalize the environment.

3. Take a look at the User Guide, especially in "Observations", "Actions", and "Rewards".

4. Experiment transforming the highway-env in a MO env.
   *In simple terms, you will make the highway-env a mo-gymnasium environment (https://mo-gymnasium.farama.org/index.html). Then check how to wrap (use a Wrapper) so you can modify the env to return multiple reward vectors. (I'll include an example for the DeepSeaTreasure, which is already MO, but gives you an idea of how it works)*

5. Define how we are going to measure the objectives, and how to set up the rewards associated with such objectives. 
   *Objectives so far: Efficiency, Safety, and Stability*

6. Define what our state and action are going to be.

7. Define the exact environment configuration/scenario that will work on. What constitutes an interesting setting for us to test?

8. Experiment with MORL baselines (e.g., PCN) – https://github.com/LucasAlegre/morl-baselines

---

## CARLA Simulator Setup
Study how CARLA works and formulate an MO version of it. 

9. You can follow the "Getting Started" guide: https://carla.readthedocs.io/en/0.9.16/start_quickstart/

   After installing it, you can run it with:
   ```bash
   ./CarlaUE4.sh
Proposal: 2/13/2026
Eden Brunner, Hao Wang
Autonomous Driving for Food Delivery
1. Introduction & Background
We propose focusing our project on benchmarking multi-objective reinforcement learning (MORL) for the task of Autonomous Driving for Food Delivery. In this task, the vehicle (agent) is to pick up an order of food and drive to the customer, balancing a three-way trade-off: it needs to drive fast to deliver the food on time (Efficiency), but it can't drive so aggressively that it crashes (Safety) or "spills the soup" by braking too hard (Stability). The agent is expected to learn all the different trade-offs so that the users can later define their preferences.
2. Project Goals & Scope
The goal of this project is to prepare an RL environment and benchmark MORL algorithms to evaluate a deep MORL algorithm being developed in parallel. We chose the domain of autonomous food delivery given our interest of learning reinforcement learning implementations and designing AI systems. The associated task involves controlling a delivery vehicle to complete deliveries while balancing multiple objectives that can conflict with each other. E.g.,:
Efficiency – Deliver food quickly to meet customer expectations.
Safety – Avoid collisions and maintain safe driving behavior.
Stability – Prevent cargo from being damaged or spilled.
The project scope includes:
Environment Development and Simulation: Use highway-env for initial experiments, and transition to CARLA for more complex and realistic driving scenarios.
Multi-Objective Agent Design: Implementation of MORL algorithms to allow the agent to learn trade-offs among efficiency, safety, and stability.
Objective Trade-Off Analysis: Study how the agent balances conflicting objectives under different scenarios and user-defined preference weights.
Benchmarks: Evaluate the agent’s performance against single-objective baselines and measure metrics such as delivery time, collisions, and cargo integrity.
Visualization: Provide visualizations and performance metrics to understand how the agent makes decisions and adapts to different delivery priorities.
The focus of the project is on research. Specifically, we will set up experiments with simulated environments, producing insights into how MORL algorithms can handle real-world multi-objective control tasks.
3. Technologies & Frameworks
Python is our primary language, given the extensive support for deep RL (PyTorch, Gymnasium, morl-baselines) and strong integration with autonomous driving environments. For the environment specifically, we considered using the simpler highway-env for rapid prototyping (Fig. 1) and the more robust CARLA for a more comprehensive testing. CARLA is an open-source autonomous driving simulator built on Unreal Engine with pretty realistic urban environments with traffic, pedestrians, and traffic signals.
(Figure 1: Highway-env)
4. Testing Plan
Unit Tests: Check that each reward (efficiency, safety, stability) and the environment setup work correctly.
Simulation Tests: Run the agent in highway-env and CARLA to see how it balances speed, safety, and stability in different situations.
Comparisons: Compare the multi-objective agent with single-objective agents to see how well it handles trade-offs.
Preference Tests: Change the user-defined weights for different objectives and see how the agent adapts.
Visualization: Track delivery time, collisions, and cargo stability, and visualize the agent’s decisions to understand its behavior.
5. Team Strengths
Eden has a background in an Autonomous Mobile Robotics course, which gives us a solid understanding of the kinematics and control problems we’ll be working with. Along with Hao’s interest in applying RL, this project is a great chance to connect classic robotics control with modern AI-based decision making.
6. Communication Plan
We'll communicate through Discord for day-to-day coordination and hold regular meetings every Wednesday with our lead PhD student, Marcelo d'Almeida.
7. Initial User Stories
As a customer ordering a delicate meal (like hot soup), I want the autonomous vehicle to limit its maximum acceleration and braking force, so that my food arrives intact without spilling.
As a delivery fleet operator, I want the vehicle to maximize its average speed on the highway without violating safety constraints, so that food is delivered while it is still hot and fresh.
As a safety compliance officer, I want the agent to maintain a safe following distance from other vehicles in the highway-env simulation, so that the risk of collisions is minimized.
As a researcher, I want to visualize the specific trade-offs the agent makes between speed and stability in real-time, so that I can analyze how the MORL algorithm resolves conflicting goals.
As a customer with a scheduled delivery time, I want the vehicle to deliver my food exactly at the agreed time, so that I can eat it fresh without waiting.
As a customer ordering food, I want the agent to choose the route that avoids traffic jams, so that my order stays fresh.
As a delivery fleet operator, I want the vehicle to adapt driving style depending on weather conditions, so that safety is always maintained without sacrificing efficiency.
As a researcher, I want to analyze how different user-defined preference weights affect agent behavior, so that I can study the flexibility of MORL in resolving conflicting objectives.
As a fleet manager, I want to set different driving preferences for different types of deliveries, so that vehicles automatically adjust to the cargo type.
As a customer in a congested city, I want the vehicle to choose routes that balance speed and stability, so that my food arrives quickly but safely over bumpy streets.
8. Project Timeline
Week 1: Environment setup & acquisition of basic knowledge regarding MORL.
Week 2: Continue work with highway-env
Week 3 - Week 6: CARLA
Week 7 - Week 8: Visualization
Week 9 - Week 10: Benchmark and Ablation of a new algorithm
9. Division of Labor
Hao Wang: Responsible for the implementation of RL algorithms, the design of the multi-objective reward function, and the development of training scripts.
Eden Brunner: Responsible for configuring the Environment Wrappers, adjusting dynamics constraint parameters (to model stability), and creating system visualizations and demos.
10. Foreseen Issues
Reward Hacking: There is a significant risk that the agent may maximize the "safety" reward by simply refusing to move. This would satisfy the safety constraint (zero collisions) but fail the primary delivery task. We must carefully balance the penalty for inaction against the reward for safety. How to deal with it: MORL can potentially mitigate this issue by having distinct reward components.
Sim-to-Real Gap: While highway-env is excellent for training efficiency, it lacks high-fidelity physics. There is a risk that policies trained in this simplified environment may not account for complex real-world dynamics, particularly regarding the fluid dynamics of the cargo ("spilling the soup"). How to deal with it: Mitigating the Sim-to-Real Gap. We bridge the gap by moving from highway-env to CARLA before making real-world claims. Unlike highway-env, CARLA provides realistic vehicle dynamics, continuous control, perception sensors, and complex traffic, making behaviors closer to real driving.
Computational Resources: We are currently relying on Google Colab for our computing needs. There is a risk that the runtime limits and disconnection issues inherent to the free or standard tiers of Colab may interrupt long training sessions, slowing down our iteration cycle. How to deal with it: Deep RL is computationally demanding and typically requires GPU acceleration for practical training times. Currently, we rely on Google Colab and any available local GPU machines, but Colab’s runtime limits and potential disconnections can interrupt long training runs and slow iteration. To address this, we have also requested access to departmental computing resources for more stable and extended experiments.
11. Architecture Details
Proposed Systems Architecture Diagram
Configuration Layer: Hyperparameters, Environment Settings.
Environment Layer: Highway-env (base), Multiple Objective Wrapper (Safety Reward, Efficiency Reward, Stability Reward), CARLA environment.
Agent Layer: Multi-Objective RL Algorithm.
Evaluation Layer: Metrics Tracking, Visualization.
12. Expected Impact
The primary impact of this project is to demonstrate the feasibility of using multi-objective reinforcement learning (MORL) to solve complex, conflicting trade-offs in autonomous logistics. Rather than optimizing for a single metric, we aim to provide a working prototype for a delivery system that dynamically balances efficiency (speed and time), safety (collision avoidance), and cargo stability. If successful, this project could serve as a benchmark for how customizable, multi-faceted constraints can be integrated into autonomous driving stacks, allowing for adaptable driving policies suited to diverse last-mile delivery scenarios.
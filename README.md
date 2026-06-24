# RocketLander

A high-fidelity 3D Rocket Landing environment for reinforcement learning built with [Gymnasium](https://gymnasium.farama.org/) and [PyBullet](https://pybullet.org/).

## Features

- **Dynamic Body Mass** - Simulation of fuel consumption and its effect on inertia and mass.
- **Phase-Based Reward Shaping** - Specialized `Glideslope` reward logic that guides agents from high-altitude approach to a soft touchdown.
- **Curriculum Learning** - Automatically scales difficulty by increasing spawn radius and randomized initial conditions (tilt, velocity).
- **Control Actuators**:
    - **Main Engine**: Gimbaled thruster with throttle control.
    - **RCS**: 12-thruster Reaction Control System for attitude stability.
    - **Landing Legs**: Deployable legs for touchdown stability.

## Installation

```bash
git clone https://github.com/fitranurmayadi/Reinforcement-Learning.git
cd Reinforcement-Learning/RocketLander
pip install -e .
```

## Quick Start (RL Visualization)

To view the current PPO agent in action:

```bash
python enjoy_rl.py --model ./models/ppo_rocket_curriculum/rocket_final.zip --stats ./models/ppo_rocket_curriculum/vec_normalize.pkl
```

## Training

The environment supports curriculum learning out of the box. To start a training session:

```bash
python train_ppo_curriculum.py
```

## Authors
- **Fitra Nurmayadi** - *Initial Work & Architecture*

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
git clone https://github.com/fitranurmayadi/RocketLander3D.git
cd RocketLander3D
pip install -e .
```

## Quick Start (RL Visualization)

To view the current PPO agent in action:

```bash
python enjoy_rl.py --model ./models/ppo_rocket_curriculum/rocket_final.zip --stats ./models/ppo_rocket_curriculum/vec_normalize.pkl
```

## Quick Start (Mission Simulator)

To run the trajectory-planned mission simulation with the RocketLander3D flight controller:

```bash
python rocketlander/run_rocketlander3d.py
```

To run it headless (no GUI):

```bash
python rocketlander/run_rocketlander3d.py --no-render
```

After the simulation completes, the performance report will be saved inside the `images/` directory as `images/ultimate_mission_report_rocketlander3d.png`.

## Training

The environment supports curriculum learning out of the box. To start a training session:

```bash
python train_ppo_curriculum.py
```

## Authors
- **Fitra Nurmayadi** - *Initial Work & Architecture*

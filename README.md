# F1-Track-Time-Machine-Learning

This project simulates an F1 race car navigating around a track using Reinforcement Learning (RL). The agent learns to control the car's steering and throttle to complete laps efficiently. The simulation includes realistic car dynamics, various track layouts, and visualizations to observe the agent's performance improvement over time.

## Table of Contents
- [Introduction](#introduction)
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Simulation](#running-the-simulation)
  - [Visualizing Performance](#visualizing-performance)
- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Visualization and Analysis](#visualization-and-analysis)
  - [Total Reward Plot](#total-reward-plot)
  - [Animations](#animations)
  - [TensorBoard Visualization](#tensorboard-visualization)
- [Additional Notes](#additional-notes)
- [License](#license)

## Introduction

Reinforcement Learning has shown great promise in teaching agents to perform complex tasks by interacting with their environment. This project demonstrates how an RL agent can learn to drive an F1 race car around a track, optimizing for speed and adherence to the track path. By incorporating realistic physics and rendering, the simulation provides an engaging way to explore RL concepts in the context of autonomous driving.

## Project Description

The project consists of:

- **Environment**: A custom OpenAI Gym environment (`F1TrackEnv`) that simulates the car and track dynamics.
- **Agent**: An RL agent trained using the Proximal Policy Optimization (PPO) algorithm from Stable Baselines 3.
- **Visualization**: Tools to render the car's movement around the track and create animations showing the agent's performance over time.

### Key features include:

- **Realistic Car Dynamics**: The car model considers physical properties like mass, drag, tire grip, and steering mechanics.
- **Track Variations**: Supports different track layouts (e.g., circular, figure-eight) with varying surface friction to simulate conditions like wet sections.
- **Performance Logging**: Logs the agent's performance data for analysis.
- **Visualization**: Animations and plots to visualize the agent's learning progress and performance improvements.

## Installation

Before running the project, ensure you have the following dependencies installed:

## Usage

### Running the Simulation
1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/F1-Track-Time-Machine-Learning.git
    cd F1-Track-Time-Machine-Learning
    ```

2. **Run the Main Script**:

    ```bash
    python f1_rl_simulation.py
    ```

This will:

- Initialize the environment and agent.
- Train the agent over a specified number of timesteps.
- Save the model at regular intervals.
- Collect performance data for visualization.

### Visualizing Performance

After training, the script will generate:

- **Total Reward Plot**: Shows the agent's total reward over time, indicating learning progress.
- **Animations**: Visual representations of the car's path around the track at different training stages.

To view the animations:

- The script will display them automatically using `matplotlib`.
- If you wish to save the animations as video files, ensure `ffmpeg` is installed and uncomment the relevant line in the `create_animation` function.

### Code Structure

- **Car Class**: Represents the car's dynamics, updating position, speed, and heading based on control inputs.
- **Track Class**: Defines the track layout, including waypoints, elevations, and surface friction profiles.
- **Logger Class**: Logs simulation data to a CSV file for analysis.
- **F1TrackEnv Class**: Custom Gym environment encapsulating the car and track, providing step and reset functions for the agent.
- **Main Script** (`if __name__ == "__main__":` block):
  - Initializes the environment and agent.
  - Trains the agent, saving models at intervals.
  - Runs evaluations of the agent at different training stages.
  - Generates plots and animations to visualize performance.

### Dependencies

- **Python 3.7+**
- **Gymnasium**: For creating custom RL environments.
- **Stable Baselines 3**: Provides the PPO algorithm for training.
- **Pygame**: Used for rendering the simulation.
- **Matplotlib**: For plotting and creating animations.
- **NumPy**: Fundamental package for scientific computing.

### Visualization and Analysis

#### Total Reward Plot

- **Purpose**: Shows how the agent's total reward changes over time.
- **Interpretation**: An increasing trend indicates that the agent is learning to perform better.

#### Animations

- **Purpose**: Visualize the car's trajectory around the track at different training stages.
- **How to View**: The animations will pop up after running the script.
- **Saving Animations**: To save animations as MP4 files, uncomment the `ani.save()` line in the `create_animation` function and ensure `ffmpeg` is installed.

#### TensorBoard Visualization

- **Setup**: Training logs are saved in the `ppo_f1_tensorboard` directory.

    **Usage**:

    ```bash
    tensorboard --logdir ./ppo_f1_tensorboard/
    ```

- **Features**: Allows real-time monitoring of training metrics like loss and reward.

### Additional Notes

- **Adjusting Parameters**:
  - `total_timesteps`: Total number of training steps.
  - `save_interval`: Frequency of saving the model and evaluating performance.
  Modify these in the main script as needed.

- **Rendering During Training**: Rendering is disabled during training to improve speed. It is enabled during evaluation runs for visualization.

- **Customizing Tracks**: You can create new track layouts by modifying the `create_track` method in the `Track` class.

- **Performance Metrics**: Additional metrics can be logged and plotted by extending the `Logger` class and updating the plotting functions.


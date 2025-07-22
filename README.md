# CartPole Reinforcement Learning Agent 🎯

This project implements a basic reinforcement learning (RL) agent using PyTorch to solve the classic CartPole-v1 environment from OpenAI Gym.

## 🧠 What is CartPole?

CartPole is a classic RL problem where the agent must balance a pole on a moving cart by applying left or right forces.

## ⚙️ Technologies Used

- Python
- PyTorch
- OpenAI Gym
- NumPy

## 🚀 How it Works

The agent uses a policy gradient method (REINFORCE) to learn:
- Take action based on current state
- Collect rewards
- Update the policy using gradient ascent

## 📁 Files

- `train.py` — main training loop
- `policy.py` — defines the neural network policy
- `utils.py` — helper functions
- `requirements.txt` — dependencies

## 🧪 To Run the Project

```bash
pip install -r requirements.txt
python train.py


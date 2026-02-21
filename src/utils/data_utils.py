"""Data processing utilities for GBWM RL"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Any


def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                dones: torch.Tensor,
                gamma: float = 0.99,
                gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)

    Args:
        rewards: Tensor of rewards (T,)
        values: Tensor of value estimates (T,)
        dones: Tensor of done flags (T,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Convert dones to float for arithmetic operations
    dones_float = dones.float()

    # Work backwards through time
    gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0  # Terminal state
        else:
            next_value = values[t + 1] * (1 - dones_float[t])

        # TD error
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE calculation
        gae = delta + gamma * gae_lambda * gae * (1 - dones_float[t])
        advantages[t] = gae

        # Compute returns
        if t == T - 1:
            returns[t] = rewards[t]
        else:
            returns[t] = rewards[t] + gamma * returns[t + 1] * (1 - dones_float[t])

    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have zero mean and unit variance"""
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def discount_rewards(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted rewards

    Args:
        rewards: List of rewards
        gamma: Discount factor

    Returns:
        List of discounted rewards
    """
    discounted = []
    cumulative = 0

    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        discounted.append(cumulative)

    return list(reversed(discounted))


def compute_returns(rewards: torch.Tensor,
                    values: torch.Tensor,
                    dones: torch.Tensor,
                    gamma: float = 0.99) -> torch.Tensor:
    """
    Compute returns using Monte Carlo method

    Args:
        rewards: Tensor of rewards
        values: Tensor of value estimates (unused in MC)
        dones: Tensor of done flags
        gamma: Discount factor

    Returns:
        Tensor of returns
    """
    T = len(rewards)
    returns = torch.zeros_like(rewards)
    dones_float = dones.float()

    for t in reversed(range(T)):
        if t == T - 1 or dones[t]:
            returns[t] = rewards[t]
        else:
            returns[t] = rewards[t] + gamma * returns[t + 1]

    return returns


def flatten_trajectories(trajectories: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
    """
    Flatten list of trajectories into batch format

    Args:
        trajectories: List of trajectory data

    Returns:
        Flattened data as tensors
    """
    states = []
    actions = []
    rewards = []
    dones = []

    for trajectory in trajectories:
        for step in trajectory:
            states.append(step['state'])
            actions.append(step['action'])
            rewards.append(step['reward'])
            dones.append(step.get('done', False))

    return {
        'states': torch.FloatTensor(states),
        'actions': torch.LongTensor(actions),
        'rewards': torch.FloatTensor(rewards),
        'dones': torch.BoolTensor(dones)
    }


def batch_generator(data: Dict[str, torch.Tensor],
                    batch_size: int,
                    shuffle: bool = True) -> Dict[str, torch.Tensor]:
    """
    Generate mini-batches from data

    Args:
        data: Dictionary of tensors
        batch_size: Size of mini-batches
        shuffle: Whether to shuffle data

    Yields:
        Mini-batch dictionaries
    """
    dataset_size = len(next(iter(data.values())))

    if shuffle:
        indices = torch.randperm(dataset_size)
    else:
        indices = torch.arange(dataset_size)

    for start in range(0, dataset_size, batch_size):
        end = min(start + batch_size, dataset_size)
        batch_indices = indices[start:end]

        batch = {}
        for key, tensor in data.items():
            batch[key] = tensor[batch_indices]

        yield batch
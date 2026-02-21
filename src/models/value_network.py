"""
Value Network (Critic) for GBWM PPO Agent

Estimates state values for advantage calculation and variance reduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from config.training_config import ModelConfig, DEFAULT_MODEL_CONFIG


class ValueNetwork(nn.Module):
    """
    Value network for estimating state values V(s)

    Takes [time, wealth] as input and outputs scalar value estimate
    representing expected future utility from current state.
    """

    def __init__(self, config: ModelConfig = None):
        super(ValueNetwork, self).__init__()

        self.config = config or DEFAULT_MODEL_CONFIG

        # Value network architecture
        self.network = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)  # Single value output
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Initialize final layer with smaller weights for value function
        nn.init.orthogonal_(self.network[-1].weight, gain=0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network

        Args:
            state: Tensor of shape (batch_size, 2) containing [time, wealth]

        Returns:
            values: Tensor of shape (batch_size,) containing value estimates
        """
        values = self.network(state).squeeze(-1)
        return values

    def predict_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict value for given state (alias for forward)

        Args:
            state: Input state tensor

        Returns:
            Predicted value
        """
        return self.forward(state)


class DualValueNetwork(nn.Module):
    """
    Alternative dual-head value network

    Predicts separate values for goal-taking and goal-skipping scenarios.
    Can provide more nuanced value estimates but increases complexity.
    """

    def __init__(self, config: ModelConfig = None):
        super(DualValueNetwork, self).__init__()

        self.config = config or DEFAULT_MODEL_CONFIG

        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU()
        )

        # Value heads
        self.goal_value_head = nn.Linear(self.config.hidden_dim, 1)
        self.skip_value_head = nn.Linear(self.config.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Smaller initialization for value heads
        nn.init.orthogonal_(self.goal_value_head.weight, gain=0.1)
        nn.init.orthogonal_(self.skip_value_head.weight, gain=0.1)

    def forward(self, state: torch.Tensor, goal_available: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with goal-aware value estimation

        Args:
            state: Input state tensor
            goal_available: Boolean tensor indicating if goal is available

        Returns:
            Value estimates
        """
        shared_features = self.shared_backbone(state)

        goal_values = self.goal_value_head(shared_features).squeeze(-1)
        skip_values = self.skip_value_head(shared_features).squeeze(-1)

        if goal_available is not None:
            # Weighted combination based on goal availability
            values = goal_available * goal_values + (1 - goal_available) * skip_values
        else:
            # Default to average of both heads
            values = (goal_values + skip_values) / 2

        return values
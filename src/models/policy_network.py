"""
Policy Network (Actor) for GBWM PPO Agent

Multi-head architecture with shared backbone for coordinated decision making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional

from config.training_config import ModelConfig, DEFAULT_MODEL_CONFIG


class PolicyNetwork(nn.Module):
    """
    Multi-head policy network for GBWM

    Architecture:
    - Shared backbone: [time, wealth] -> shared features
    - Goal head: shared features -> [skip_prob, take_prob]
    - Portfolio head: shared features -> [portfolio_1_prob, ..., portfolio_15_prob]
    """

    def __init__(self, config: ModelConfig = None):
        super(PolicyNetwork, self).__init__()

        self.config = config or DEFAULT_MODEL_CONFIG

        # Shared backbone layers
        self.shared_backbone = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU()
        )

        # Goal decision head (binary: skip or take)
        self.goal_head = nn.Linear(self.config.hidden_dim, self.config.goal_action_dim)

        # Portfolio selection head (categorical: 15 portfolios)
        self.portfolio_head = nn.Linear(self.config.hidden_dim, self.config.portfolio_action_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network

        Args:
            state: Tensor of shape (batch_size, 2) containing [time, wealth]

        Returns:
            Tuple of (goal_probs, portfolio_probs)
            - goal_probs: (batch_size, 2) - probabilities for [skip, take]
            - portfolio_probs: (batch_size, 15) - probabilities for each portfolio
        """
        # Extract shared features
        shared_features = self.shared_backbone(state)

        # Goal decision probabilities
        goal_logits = self.goal_head(shared_features)
        goal_probs = F.softmax(goal_logits, dim=-1)

        # Portfolio selection probabilities
        portfolio_logits = self.portfolio_head(shared_features)
        portfolio_probs = F.softmax(portfolio_logits, dim=-1)

        return goal_probs, portfolio_probs

    def get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Sample actions and compute log probabilities

        Args:
            state: Input state tensor
            deterministic: If True, take most probable actions (for evaluation)

        Returns:
            Tuple of (actions, log_probs)
            - actions: (batch_size, 2) - [goal_action, portfolio_action]
            - log_probs: (batch_size,) - log probabilities of taken actions
        """
        goal_probs, portfolio_probs = self.forward(state)

        # Create categorical distributions
        goal_dist = Categorical(goal_probs)
        portfolio_dist = Categorical(portfolio_probs)

        if deterministic:
            # Take most probable actions (for evaluation)
            goal_action = torch.argmax(goal_probs, dim=-1)
            portfolio_action = torch.argmax(portfolio_probs, dim=-1)
        else:
            # Sample actions stochastically (for training)
            goal_action = goal_dist.sample()
            portfolio_action = portfolio_dist.sample()

        # Compute log probabilities
        goal_log_prob = goal_dist.log_prob(goal_action)
        portfolio_log_prob = portfolio_dist.log_prob(portfolio_action)

        # Combine actions and log probabilities
        actions = torch.stack([goal_action, portfolio_action], dim=-1)
        log_probs = goal_log_prob + portfolio_log_prob

        return actions, log_probs

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions (for PPO training)

        Args:
            state: Input states
            actions: Actions to evaluate [goal_actions, portfolio_actions]

        Returns:
            Tuple of (log_probs, entropy)
        """
        goal_probs, portfolio_probs = self.forward(state)

        # Create distributions
        goal_dist = Categorical(goal_probs)
        portfolio_dist = Categorical(portfolio_probs)

        # Extract individual actions
        goal_actions = actions[:, 0]
        portfolio_actions = actions[:, 1]

        # Compute log probabilities
        goal_log_probs = goal_dist.log_prob(goal_actions)
        portfolio_log_probs = portfolio_dist.log_prob(portfolio_actions)

        # Compute entropy (for exploration bonus)
        goal_entropy = goal_dist.entropy()
        portfolio_entropy = portfolio_dist.entropy()

        # Combine
        log_probs = goal_log_probs + portfolio_log_probs
        entropy = goal_entropy + portfolio_entropy

        return log_probs, entropy


class PolicyNetworkLegacy(nn.Module):
    """
    Alternative single-head policy network (for comparison)
    Treats the action space as a single 30-dimensional categorical distribution
    """

    def __init__(self, config: ModelConfig = None):
        super(PolicyNetworkLegacy, self).__init__()

        self.config = config or DEFAULT_MODEL_CONFIG

        # Total action space size: 2 goal actions × 15 portfolios = 30
        self.total_action_dim = self.config.goal_action_dim * self.config.portfolio_action_dim

        self.network = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.total_action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action probabilities"""
        logits = self.network(state)
        return F.softmax(logits, dim=-1)

    def get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability"""
        action_probs = self.forward(state)
        dist = Categorical(action_probs)

        if deterministic:
            action_idx = torch.argmax(action_probs, dim=-1)
        else:
            action_idx = dist.sample()

        # Convert flat action index to [goal, portfolio] format
        goal_action = action_idx // self.config.portfolio_action_dim
        portfolio_action = action_idx % self.config.portfolio_action_dim

        actions = torch.stack([goal_action, portfolio_action], dim=-1)
        log_probs = dist.log_prob(action_idx)

        return actions, log_probs
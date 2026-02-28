"""
Sentiment-Aware Policy Network for GBWM

This module implements sentiment-aware actor networks that can process
both traditional state (time, wealth) and market sentiment features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional, Union, Dict

from .feature_encoders import create_encoder


class SentimentAwarePolicyNetwork(nn.Module):
    """
    Sentiment-aware policy network for GBWM
    
    Multi-head architecture that processes sentiment-augmented states:
    - State: [time, wealth, vix_sentiment, vix_momentum] 
    - Actions: [goal_decision, portfolio_choice]
    
    Architecture:
        State Encoder → Shared Features → Goal Head + Portfolio Head
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        num_portfolios: int = 15,
        hidden_dim: int = 64,
        encoder_type: str = "feature",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0
    ):
        """
        Initialize sentiment-aware policy network
        
        Args:
            state_dim: Input state dimensionality (2 for baseline, 4 for sentiment)
            num_portfolios: Number of portfolio choices
            hidden_dim: Hidden layer dimension
            encoder_type: Type of encoder ('feature', 'simple', 'adaptive', 'attention')
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 to disable)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_portfolios = num_portfolios
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        
        # State encoder
        self.state_encoder = create_encoder(
            encoder_type=encoder_type,
            input_dim=state_dim,
            hidden_dim=hidden_dim
        )

        # Get actual encoder output dimension (may differ from hidden_dim)
        encoder_output_dim = self.state_encoder.get_output_dim()

        # Shared feature processing - use encoder's output dimension
        shared_layers = [nn.Linear(encoder_output_dim, hidden_dim)]
        if use_batch_norm:
            shared_layers.append(nn.BatchNorm1d(hidden_dim))
        shared_layers.append(nn.Tanh())
        if dropout_rate > 0:
            shared_layers.append(nn.Dropout(dropout_rate))
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Action heads
        self.goal_head = nn.Linear(hidden_dim, 2)  # [skip, take]
        self.portfolio_head = nn.Linear(hidden_dim, num_portfolios)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for PPO training"""
        # Shared layers
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        # Action heads - small initialization for exploration
        nn.init.orthogonal_(self.goal_head.weight, gain=0.01)
        nn.init.constant_(self.goal_head.bias, 0.0)
        
        nn.init.orthogonal_(self.portfolio_head.weight, gain=0.01)
        nn.init.constant_(self.portfolio_head.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network
        
        Args:
            state: (batch_size, state_dim) state tensor
            
        Returns:
            Tuple of (goal_probs, portfolio_probs)
            - goal_probs: (batch_size, 2) probabilities for [skip, take]
            - portfolio_probs: (batch_size, num_portfolios) portfolio probabilities
        """
        # Encode state features
        encoded_state = self.state_encoder(state)
        
        # Process through shared layers
        shared_features = self.shared_layers(encoded_state)
        
        # Generate action probabilities
        goal_logits = self.goal_head(shared_features)
        portfolio_logits = self.portfolio_head(shared_features)
        
        # Apply softmax
        goal_probs = F.softmax(goal_logits, dim=-1)
        portfolio_probs = F.softmax(portfolio_logits, dim=-1)
        
        return goal_probs, portfolio_probs
    
    def get_action_and_log_prob(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions and compute log probabilities
        
        Args:
            state: Input state tensor
            deterministic: If True, take most probable actions
            
        Returns:
            Tuple of (actions, log_probs)
            - actions: (batch_size, 2) [goal_action, portfolio_action]
            - log_probs: (batch_size,) log probabilities
        """
        # Handle single state input
        single_input = (state.dim() == 1)
        if single_input:
            state = state.unsqueeze(0)
        
        goal_probs, portfolio_probs = self.forward(state)
        
        # Create categorical distributions
        goal_dist = Categorical(goal_probs)
        portfolio_dist = Categorical(portfolio_probs)
        
        if deterministic:
            # Take most probable actions
            goal_action = torch.argmax(goal_probs, dim=-1)
            portfolio_action = torch.argmax(portfolio_probs, dim=-1)
        else:
            # Sample stochastically
            goal_action = goal_dist.sample()
            portfolio_action = portfolio_dist.sample()
        
        # Compute log probabilities
        goal_log_prob = goal_dist.log_prob(goal_action)
        portfolio_log_prob = portfolio_dist.log_prob(portfolio_action)
        
        # Combine actions and log probabilities
        actions = torch.stack([goal_action, portfolio_action], dim=-1)
        log_probs = goal_log_prob + portfolio_log_prob
        
        # Handle single input case
        if single_input:
            actions = actions.squeeze(0)
            log_probs = log_probs.squeeze(0)
        
        return actions, log_probs
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions
        
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
        goal_actions = actions[:, 0].long()
        portfolio_actions = actions[:, 1].long()
        
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
    
    def get_action_probabilities(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed action probabilities for analysis
        
        Args:
            state: Input state tensor
            
        Returns:
            Dictionary with goal and portfolio probabilities
        """
        goal_probs, portfolio_probs = self.forward(state)
        
        return {
            'goal_probs': goal_probs,
            'portfolio_probs': portfolio_probs,
            'goal_skip_prob': goal_probs[:, 0],
            'goal_take_prob': goal_probs[:, 1],
            'most_likely_portfolio': torch.argmax(portfolio_probs, dim=-1)
        }


class HierarchicalPolicyNetwork(nn.Module):
    """
    Hierarchical policy network with separate sentiment processing
    
    Uses a two-level hierarchy:
    1. High-level policy: Considers sentiment for strategic decisions
    2. Low-level policy: Focuses on tactical portfolio selection
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        num_portfolios: int = 15,
        hidden_dim: int = 64
    ):
        """
        Initialize hierarchical policy network
        
        Args:
            state_dim: Input state dimensionality
            num_portfolios: Number of portfolio choices
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_portfolios = num_portfolios
        self.hidden_dim = hidden_dim
        
        # High-level encoder (focuses on sentiment + time)
        self.high_level_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh()
        )
        
        # Low-level encoder (focuses on wealth + portfolio characteristics)
        self.low_level_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh()
        )
        
        # Goal decision head (high-level)
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.Tanh(),
            nn.Linear(hidden_dim//4, 2)
        )
        
        # Portfolio selection head (low-level + context from high-level)
        self.portfolio_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),  # Combined features
            nn.Tanh(),
            nn.Linear(hidden_dim//2, num_portfolios)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through hierarchical network"""
        # High-level processing (strategic)
        high_features = self.high_level_encoder(state)
        
        # Low-level processing (tactical) 
        low_features = self.low_level_encoder(state)
        
        # Goal decision (high-level)
        goal_logits = self.goal_head(high_features)
        goal_probs = F.softmax(goal_logits, dim=-1)
        
        # Portfolio decision (combined features)
        combined_features = torch.cat([high_features, low_features], dim=-1)
        portfolio_logits = self.portfolio_head(combined_features)
        portfolio_probs = F.softmax(portfolio_logits, dim=-1)
        
        return goal_probs, portfolio_probs
    
    def get_action_and_log_prob(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from hierarchical policy"""
        single_input = (state.dim() == 1)
        if single_input:
            state = state.unsqueeze(0)
        
        goal_probs, portfolio_probs = self.forward(state)
        
        goal_dist = Categorical(goal_probs)
        portfolio_dist = Categorical(portfolio_probs)
        
        if deterministic:
            goal_action = torch.argmax(goal_probs, dim=-1)
            portfolio_action = torch.argmax(portfolio_probs, dim=-1)
        else:
            goal_action = goal_dist.sample()
            portfolio_action = portfolio_dist.sample()
        
        goal_log_prob = goal_dist.log_prob(goal_action)
        portfolio_log_prob = portfolio_dist.log_prob(portfolio_action)
        
        actions = torch.stack([goal_action, portfolio_action], dim=-1)
        log_probs = goal_log_prob + portfolio_log_prob
        
        if single_input:
            actions = actions.squeeze(0)
            log_probs = log_probs.squeeze(0)
        
        return actions, log_probs
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO training"""
        goal_probs, portfolio_probs = self.forward(state)
        
        goal_dist = Categorical(goal_probs)
        portfolio_dist = Categorical(portfolio_probs)
        
        goal_actions = actions[:, 0].long()
        portfolio_actions = actions[:, 1].long()
        
        goal_log_probs = goal_dist.log_prob(goal_actions)
        portfolio_log_probs = portfolio_dist.log_prob(portfolio_actions)
        
        goal_entropy = goal_dist.entropy()
        portfolio_entropy = portfolio_dist.entropy()
        
        log_probs = goal_log_probs + portfolio_log_probs
        entropy = goal_entropy + portfolio_entropy
        
        return log_probs, entropy


def create_sentiment_policy(
    policy_type: str = "standard",
    state_dim: int = 4,
    num_portfolios: int = 15,
    **kwargs
) -> nn.Module:
    """
    Factory function to create sentiment-aware policy networks
    
    Args:
        policy_type: Type of policy ('standard', 'hierarchical')
        state_dim: Input state dimensionality
        num_portfolios: Number of portfolio choices
        **kwargs: Additional policy parameters
        
    Returns:
        Initialized policy network
    """
    if policy_type == "standard":
        return SentimentAwarePolicyNetwork(
            state_dim=state_dim,
            num_portfolios=num_portfolios,
            **kwargs
        )
    elif policy_type == "hierarchical":
        # HierarchicalPolicyNetwork has its own internal encoding, so filter out encoder_type
        hierarchical_kwargs = {k: v for k, v in kwargs.items() if k != 'encoder_type'}
        return HierarchicalPolicyNetwork(
            state_dim=state_dim,
            num_portfolios=num_portfolios,
            **hierarchical_kwargs
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def test_sentiment_policies():
    """Test function for sentiment-aware policy networks"""
    print("Testing sentiment-aware policy networks...")
    
    batch_size = 32
    state_2d = torch.randn(batch_size, 2)  
    state_4d = torch.randn(batch_size, 4)  
    
    try:
        # Test standard sentiment-aware policy
        policy = SentimentAwarePolicyNetwork(state_dim=4, num_portfolios=15)
        
        # Test forward pass
        goal_probs, portfolio_probs = policy.forward(state_4d)
        assert goal_probs.shape == (batch_size, 2), f"Wrong goal probs shape: {goal_probs.shape}"
        assert portfolio_probs.shape == (batch_size, 15), f"Wrong portfolio probs shape: {portfolio_probs.shape}"
        print("✓ Standard policy forward pass test passed")
        
        # Test action sampling
        actions, log_probs = policy.get_action_and_log_prob(state_4d)
        assert actions.shape == (batch_size, 2), f"Wrong actions shape: {actions.shape}"
        assert log_probs.shape == (batch_size,), f"Wrong log_probs shape: {log_probs.shape}"
        print("✓ Standard policy action sampling test passed")
        
        # Test action evaluation
        eval_log_probs, entropy = policy.evaluate_actions(state_4d, actions)
        assert eval_log_probs.shape == (batch_size,), f"Wrong eval log_probs shape: {eval_log_probs.shape}"
        assert entropy.shape == (batch_size,), f"Wrong entropy shape: {entropy.shape}"
        print("✓ Standard policy action evaluation test passed")
        
        # Test single state input
        single_state = state_4d[0]  # Shape: (4,)
        single_action, single_log_prob = policy.get_action_and_log_prob(single_state)
        assert single_action.shape == (2,), f"Wrong single action shape: {single_action.shape}"
        assert single_log_prob.shape == (), f"Wrong single log_prob shape: {single_log_prob.shape}"
        print("✓ Single state input test passed")
        
        # Test hierarchical policy
        hierarchical_policy = HierarchicalPolicyNetwork(state_dim=4, num_portfolios=15)
        
        actions_h, log_probs_h = hierarchical_policy.get_action_and_log_prob(state_4d)
        assert actions_h.shape == (batch_size, 2), f"Wrong hierarchical actions shape: {actions_h.shape}"
        print("✓ Hierarchical policy test passed")
        
        # Test adaptive state dimensions
        policy_adaptive = SentimentAwarePolicyNetwork(
            state_dim=2,
            encoder_type="simple",
            num_portfolios=15
        )
        
        actions_2d, _ = policy_adaptive.get_action_and_log_prob(state_2d)
        assert actions_2d.shape == (batch_size, 2), f"Wrong 2D actions shape: {actions_2d.shape}"
        print("✓ Adaptive state dimensions test passed")
        
        # Test factory function
        policies = [
            create_sentiment_policy("standard", state_dim=4),
            create_sentiment_policy("hierarchical", state_dim=4)
        ]
        print("✓ Policy factory test passed")
        
        print("All sentiment policy tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"✗ Sentiment policy test failed: {e}")
        return False


if __name__ == "__main__":
    test_sentiment_policies()
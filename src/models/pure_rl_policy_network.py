"""
Pure RL Policy Network with Advanced Architectures for GBWM

This module implements Pure RL policy networks with the same architectural
options as Sentiment RL, but for 2D state [time, wealth] without VIX features.

Available architectures:
- Standard: encoder → shared layers → goal/portfolio heads
- Hierarchical: high-level (goal) + low-level (portfolio) networks
- Encoder options: simple, attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Dict, Optional


class SimpleEncoder2D(nn.Module):
    """
    Simple encoder for 2D state [time, wealth]

    Two-layer MLP with Tanh activations, matching the structure used
    in Sentiment RL's SimpleEncoder.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)


class AttentionEncoder2D(nn.Module):
    """
    Attention-based encoder for 2D state

    Uses self-attention to learn feature importance between time and wealth.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_heads: int = 2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project input to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Project to hidden dimension
        projected = self.input_projection(state)  # (batch_size, hidden_dim)

        # Add sequence dimension for attention
        seq_input = projected.unsqueeze(1)

        # Self-attention
        attn_output, _ = self.self_attention(seq_input, seq_input, seq_input)
        attn_output = attn_output.squeeze(1)

        # Residual connection + layer norm
        normed_attn = self.layer_norm(projected + attn_output)

        # Feed-forward
        ff_output = self.feed_forward(normed_attn)

        # Final residual + norm
        encoded = self.final_norm(normed_attn + ff_output)

        return encoded


def create_encoder_2d(
    encoder_type: str = "simple",
    input_dim: int = 2,
    hidden_dim: int = 64,
    **kwargs
) -> nn.Module:
    """
    Factory function to create 2D state encoders

    Args:
        encoder_type: Type of encoder ('simple', 'attention')
        input_dim: Input dimensionality (default: 2)
        hidden_dim: Hidden layer dimensionality
        **kwargs: Additional encoder parameters

    Returns:
        Initialized encoder module
    """
    if encoder_type == "simple":
        return SimpleEncoder2D(input_dim=input_dim, hidden_dim=hidden_dim)
    elif encoder_type == "attention":
        return AttentionEncoder2D(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type for 2D: {encoder_type}. Use 'simple' or 'attention'.")


class PureRLPolicyNetwork(nn.Module):
    """
    Pure RL policy network with advanced architecture options

    Multi-head architecture for 2D state [time, wealth]:
    - State Encoder → Shared Features → Goal Head + Portfolio Head

    This mirrors SentimentAwarePolicyNetwork but for 2D state.
    """

    def __init__(
        self,
        state_dim: int = 2,
        num_portfolios: int = 15,
        hidden_dim: int = 64,
        encoder_type: str = "simple",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0
    ):
        """
        Initialize Pure RL policy network

        Args:
            state_dim: Input state dimensionality (default: 2 for [time, wealth])
            num_portfolios: Number of portfolio choices
            hidden_dim: Hidden layer dimension
            encoder_type: Type of encoder ('simple', 'attention')
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 to disable)
        """
        super().__init__()

        self.state_dim = state_dim
        self.num_portfolios = num_portfolios
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type

        # State encoder
        self.state_encoder = create_encoder_2d(
            encoder_type=encoder_type,
            input_dim=state_dim,
            hidden_dim=hidden_dim
        )

        encoder_output_dim = self.state_encoder.get_output_dim()

        # Shared feature processing
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

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for PPO training"""
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
        """
        encoded_state = self.state_encoder(state)
        shared_features = self.shared_layers(encoded_state)

        goal_logits = self.goal_head(shared_features)
        portfolio_logits = self.portfolio_head(shared_features)

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
        """
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
        """
        Evaluate log probabilities and entropy for given actions

        Args:
            state: Input states
            actions: Actions to evaluate [goal_actions, portfolio_actions]

        Returns:
            Tuple of (log_probs, entropy)
        """
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

    def get_action_probabilities(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed action probabilities for analysis"""
        goal_probs, portfolio_probs = self.forward(state)

        return {
            'goal_probs': goal_probs,
            'portfolio_probs': portfolio_probs,
            'goal_skip_prob': goal_probs[:, 0],
            'goal_take_prob': goal_probs[:, 1],
            'most_likely_portfolio': torch.argmax(portfolio_probs, dim=-1)
        }


class HierarchicalPolicyNetwork2D(nn.Module):
    """
    Hierarchical policy network for 2D state

    Uses a two-level hierarchy:
    1. High-level policy: Strategic goal decisions based on time
    2. Low-level policy: Tactical portfolio selection based on wealth
    """

    def __init__(
        self,
        state_dim: int = 2,
        num_portfolios: int = 15,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_portfolios = num_portfolios
        self.hidden_dim = hidden_dim

        # High-level encoder (focuses on time for strategic decisions)
        self.high_level_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )

        # Low-level encoder (focuses on wealth for tactical decisions)
        self.low_level_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )

        # Goal decision head (high-level)
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 2)
        )

        # Portfolio selection head (low-level + context from high-level)
        self.portfolio_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # Combined features
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_portfolios)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through hierarchical network"""
        high_features = self.high_level_encoder(state)
        low_features = self.low_level_encoder(state)

        goal_logits = self.goal_head(high_features)
        goal_probs = F.softmax(goal_logits, dim=-1)

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


def create_pure_rl_policy(
    policy_type: str = "standard",
    state_dim: int = 2,
    num_portfolios: int = 15,
    **kwargs
) -> nn.Module:
    """
    Factory function to create Pure RL policy networks

    Args:
        policy_type: Type of policy ('standard', 'hierarchical')
        state_dim: Input state dimensionality (default: 2)
        num_portfolios: Number of portfolio choices
        **kwargs: Additional policy parameters

    Returns:
        Initialized policy network
    """
    if policy_type == "standard":
        return PureRLPolicyNetwork(
            state_dim=state_dim,
            num_portfolios=num_portfolios,
            **kwargs
        )
    elif policy_type == "hierarchical":
        # Filter out unsupported kwargs for hierarchical policy
        supported_kwargs = {'hidden_dim'}
        hierarchical_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}
        return HierarchicalPolicyNetwork2D(
            state_dim=state_dim,
            num_portfolios=num_portfolios,
            **hierarchical_kwargs
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Use 'standard' or 'hierarchical'.")


def test_pure_rl_policies():
    """Test function for Pure RL policy networks"""
    print("Testing Pure RL policy networks (2D state)...")

    batch_size = 32
    state_2d = torch.randn(batch_size, 2)  # [time, wealth]

    try:
        # Test standard policy with simple encoder
        policy_simple = PureRLPolicyNetwork(
            state_dim=2,
            num_portfolios=15,
            encoder_type="simple"
        )

        goal_probs, portfolio_probs = policy_simple.forward(state_2d)
        assert goal_probs.shape == (batch_size, 2), f"Wrong goal probs shape: {goal_probs.shape}"
        assert portfolio_probs.shape == (batch_size, 15), f"Wrong portfolio probs shape: {portfolio_probs.shape}"
        print("✓ Standard policy (simple encoder) forward pass test passed")

        actions, log_probs = policy_simple.get_action_and_log_prob(state_2d)
        assert actions.shape == (batch_size, 2), f"Wrong actions shape: {actions.shape}"
        assert log_probs.shape == (batch_size,), f"Wrong log_probs shape: {log_probs.shape}"
        print("✓ Standard policy action sampling test passed")

        eval_log_probs, entropy = policy_simple.evaluate_actions(state_2d, actions)
        assert eval_log_probs.shape == (batch_size,), f"Wrong eval log_probs shape: {eval_log_probs.shape}"
        assert entropy.shape == (batch_size,), f"Wrong entropy shape: {entropy.shape}"
        print("✓ Standard policy action evaluation test passed")

        # Test standard policy with attention encoder
        policy_attention = PureRLPolicyNetwork(
            state_dim=2,
            num_portfolios=15,
            encoder_type="attention"
        )

        actions_att, log_probs_att = policy_attention.get_action_and_log_prob(state_2d)
        assert actions_att.shape == (batch_size, 2), f"Wrong attention actions shape: {actions_att.shape}"
        print("✓ Standard policy (attention encoder) test passed")

        # Test hierarchical policy
        policy_hier = HierarchicalPolicyNetwork2D(
            state_dim=2,
            num_portfolios=15
        )

        actions_hier, log_probs_hier = policy_hier.get_action_and_log_prob(state_2d)
        assert actions_hier.shape == (batch_size, 2), f"Wrong hierarchical actions shape: {actions_hier.shape}"
        print("✓ Hierarchical policy test passed")

        # Test single state input
        single_state = state_2d[0]  # Shape: (2,)
        single_action, single_log_prob = policy_simple.get_action_and_log_prob(single_state)
        assert single_action.shape == (2,), f"Wrong single action shape: {single_action.shape}"
        assert single_log_prob.shape == (), f"Wrong single log_prob shape: {single_log_prob.shape}"
        print("✓ Single state input test passed")

        # Test factory function
        policies = [
            create_pure_rl_policy("standard", encoder_type="simple"),
            create_pure_rl_policy("standard", encoder_type="attention"),
            create_pure_rl_policy("hierarchical")
        ]
        print("✓ Policy factory test passed")

        print("\nAll Pure RL policy tests passed! ✓")
        return True

    except Exception as e:
        print(f"✗ Pure RL policy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_pure_rl_policies()
"""Package initialization file"""
"""Models package for GBWM RL implementation"""

from .policy_network import PolicyNetwork, PolicyNetworkLegacy
from .value_network import ValueNetwork, DualValueNetwork

# Pure RL with advanced architectures
from .pure_rl_policy_network import (
    PureRLPolicyNetwork,
    HierarchicalPolicyNetwork2D,
    create_pure_rl_policy,
    SimpleEncoder2D,
    AttentionEncoder2D
)
from .pure_rl_value_network import (
    PureRLValueNetwork,
    DualHeadValueNetwork2D,
    EnsembleValueNetwork2D,
    create_pure_rl_value_network
)
from .pure_rl_ppo_agent import PureRLPPOAgent, PureRLAgentConfig

__all__ = [
    # Original networks
    'PolicyNetwork',
    'PolicyNetworkLegacy',
    'ValueNetwork',
    'DualValueNetwork',
    # Pure RL with advanced architectures
    'PureRLPolicyNetwork',
    'HierarchicalPolicyNetwork2D',
    'create_pure_rl_policy',
    'SimpleEncoder2D',
    'AttentionEncoder2D',
    'PureRLValueNetwork',
    'DualHeadValueNetwork2D',
    'EnsembleValueNetwork2D',
    'create_pure_rl_value_network',
    'PureRLPPOAgent',
    'PureRLAgentConfig',
]
"""
Sentiment-Aware PPO Agent for GBWM

This module implements a PPO agent that can handle sentiment-augmented states
and provides comprehensive training capabilities for the GBWM system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import deque
from pathlib import Path
import json

from .sentiment_policy_network import SentimentAwarePolicyNetwork, create_sentiment_policy
from .sentiment_value_network import SentimentAwareValueNetwork, create_sentiment_value_network
from config.training_config import TrainingConfig, DEFAULT_TRAINING_CONFIG
from src.utils.data_utils import compute_gae


class SentimentAwarePPOAgent:
    """
    Sentiment-aware PPO Agent for Goals-Based Wealth Management
    
    Extends the standard PPO agent to handle sentiment-augmented states:
    - State: [time, wealth, vix_sentiment, vix_momentum]
    - Supports both sentiment and baseline modes
    - Enhanced logging with sentiment analytics
    """
    
    def __init__(
        self,
        env,
        config: TrainingConfig = None,
        policy_type: str = "standard",
        value_type: str = "standard",
        encoder_type: str = "feature",
        device: str = None,
        sentiment_enabled: bool = True
    ):
        """
        Initialize sentiment-aware PPO agent
        
        Args:
            env: GBWM environment (sentiment-augmented or standard)
            config: Training configuration
            policy_type: Type of policy network ('standard', 'hierarchical')
            value_type: Type of value network ('standard', 'dual_head', 'ensemble')
            encoder_type: Type of state encoder ('feature', 'simple', 'adaptive', 'attention')
            device: Device to run on ('cuda' or 'cpu')
            sentiment_enabled: Whether sentiment features are enabled
        """
        self.env = env
        self.config = config or DEFAULT_TRAINING_CONFIG
        self.device = device or self.config.device
        self.sentiment_enabled = sentiment_enabled
        
        # Determine state dimensionality
        if hasattr(env, 'observation_space'):
            self.state_dim = env.observation_space.shape[0]
        else:
            # Fallback for custom environments
            self.state_dim = 4 if sentiment_enabled else 2
        
        # Determine number of portfolios
        if hasattr(env, 'action_space'):
            self.num_portfolios = env.action_space.nvec[1]  # MultiDiscrete action space
        else:
            self.num_portfolios = 15  # Default
        
        # Initialize networks
        self.policy_net = create_sentiment_policy(
            policy_type=policy_type,
            state_dim=self.state_dim,
            num_portfolios=self.num_portfolios,
            encoder_type=encoder_type,
            hidden_dim=self.config.n_neurons
        ).to(self.device)
        
        self.value_net = create_sentiment_value_network(
            network_type=value_type,
            state_dim=self.state_dim,
            encoder_type=encoder_type,
            hidden_dim=self.config.n_neurons
        ).to(self.device)
        
        # Initialize optimizers with appropriate learning rates
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )
        
        # Learning rate schedulers
        total_updates = self.config.n_traj // self.config.batch_size
        self.policy_scheduler = optim.lr_scheduler.LinearLR(
            self.policy_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_updates
        )
        self.value_scheduler = optim.lr_scheduler.LinearLR(
            self.value_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_updates
        )
        
        # Enhanced training metrics with sentiment tracking
        self.training_metrics = {
            'policy_losses': deque(maxlen=100),
            'value_losses': deque(maxlen=100),
            'total_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'advantages': deque(maxlen=1000),
            'goal_success_rates': deque(maxlen=100),
            'portfolio_selections': deque(maxlen=1000),
        }
        
        # Sentiment-specific metrics
        if sentiment_enabled:
            self.training_metrics.update({
                'vix_sentiment_values': deque(maxlen=1000),
                'vix_momentum_values': deque(maxlen=1000),
                'sentiment_correlation_rewards': deque(maxlen=100),
                'high_vix_decisions': deque(maxlen=100),
                'low_vix_decisions': deque(maxlen=100)
            })
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.total_timesteps = 0
        self.total_episodes = 0
        self.iteration = 0
        
        # Configuration tracking
        self.agent_config = {
            'policy_type': policy_type,
            'value_type': value_type,
            'encoder_type': encoder_type,
            'sentiment_enabled': sentiment_enabled,
            'state_dim': int(self.state_dim),  # Convert to Python int for JSON serialization
            'num_portfolios': int(self.num_portfolios)  # Convert to Python int for JSON serialization
        }
        
        self.logger.info(
            f"SentimentAwarePPOAgent initialized: "
            f"state_dim={self.state_dim}, "
            f"sentiment_enabled={sentiment_enabled}, "
            f"policy_type={policy_type}, "
            f"encoder_type={encoder_type}"
        )
    
    def collect_trajectories(self, num_trajectories: int) -> Dict[str, torch.Tensor]:
        """
        Collect batch of trajectories with sentiment tracking
        
        Args:
            num_trajectories: Number of trajectories to collect
            
        Returns:
            Dictionary containing collected data with sentiment information
        """
        trajectories = []
        
        self.policy_net.eval()
        self.value_net.eval()
        
        # Sentiment tracking for this batch
        batch_sentiment_data = {
            'vix_sentiment_values': [],
            'vix_momentum_values': [],
            'goal_decisions': [],
            'portfolio_decisions': [],
            'rewards': [],
            'high_vix_episodes': 0,
            'low_vix_episodes': 0
        }
        
        with torch.no_grad():
            for traj_idx in range(num_trajectories):
                trajectory = []
                episode_sentiment_data = {
                    'sentiments': [],
                    'rewards': [],
                    'goal_decisions': [],
                    'portfolio_decisions': []
                }
                
                # Reset environment
                obs, info = self.env.reset()
                
                # Track episode-level sentiment statistics
                episode_high_vix_count = 0
                episode_low_vix_count = 0
                
                for step in range(self.config.time_horizon):
                    # Convert observation to tensor
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    
                    # Extract sentiment features if available
                    sentiment_features = None
                    if self.sentiment_enabled and len(obs) >= 4:
                        sentiment_features = obs[2:4]  # [vix_sentiment, vix_momentum]
                        batch_sentiment_data['vix_sentiment_values'].append(sentiment_features[0])
                        batch_sentiment_data['vix_momentum_values'].append(sentiment_features[1])
                        episode_sentiment_data['sentiments'].append(sentiment_features)
                        
                        # Track VIX regimes
                        if sentiment_features[0] > 0.3:  # High fear
                            episode_high_vix_count += 1
                        elif sentiment_features[0] < -0.3:  # Low fear (complacency)
                            episode_low_vix_count += 1
                    
                    # Get action and log probability
                    actions, log_probs = self.policy_net.get_action_and_log_prob(state_tensor)
                    
                    # Get value estimate
                    values = self.value_net(state_tensor)
                    
                    # Execute action in environment
                    action_np = actions.squeeze(0).cpu().detach().numpy()
                    next_obs, reward, terminated, truncated, step_info = self.env.step(action_np)
                    
                    # Track decisions
                    goal_decision = int(action_np[0])
                    portfolio_decision = int(action_np[1])
                    
                    batch_sentiment_data['goal_decisions'].append(goal_decision)
                    batch_sentiment_data['portfolio_decisions'].append(portfolio_decision)
                    batch_sentiment_data['rewards'].append(reward)
                    
                    episode_sentiment_data['goal_decisions'].append(goal_decision)
                    episode_sentiment_data['portfolio_decisions'].append(portfolio_decision)
                    episode_sentiment_data['rewards'].append(reward)
                    
                    # Store step data
                    step_data = {
                        'state': obs.copy(),
                        'action': actions.squeeze(0).cpu(),
                        'reward': reward,
                        'log_prob': log_probs.squeeze(0).cpu(),
                        'value': values.squeeze(0).cpu(),
                        'done': terminated or truncated,
                        'info': step_info
                    }
                    
                    # Add sentiment data to step
                    if sentiment_features is not None:
                        step_data['sentiment_features'] = sentiment_features
                    
                    trajectory.append(step_data)
                    
                    obs = next_obs
                    self.total_timesteps += 1
                    
                    if terminated or truncated:
                        break
                
                trajectories.append(trajectory)
                self.total_episodes += 1
                
                # Track episode metrics
                episode_reward = sum(step['reward'] for step in trajectory)
                episode_length = len(trajectory)
                
                self.training_metrics['total_rewards'].append(episode_reward)
                self.training_metrics['episode_lengths'].append(episode_length)
                
                # Goal success rate
                goals_taken = sum(1 for step in trajectory 
                                if step.get('info', {}).get('goal_taken', False))
                goals_available = sum(1 for step in trajectory 
                                    if step.get('info', {}).get('goal_available', False))
                
                goal_success_rate = goals_taken / max(goals_available, 1)
                self.training_metrics['goal_success_rates'].append(goal_success_rate)
                
                # Sentiment-specific episode analysis
                if self.sentiment_enabled and episode_sentiment_data['sentiments']:
                    # Track episodes with high/low VIX exposure
                    if episode_high_vix_count > episode_length * 0.3:
                        batch_sentiment_data['high_vix_episodes'] += 1
                    if episode_low_vix_count > episode_length * 0.3:
                        batch_sentiment_data['low_vix_episodes'] += 1
                    
                    # Correlation between sentiment and rewards
                    if len(episode_sentiment_data['sentiments']) > 1:
                        sentiments = np.array(episode_sentiment_data['sentiments'])
                        rewards = np.array(episode_sentiment_data['rewards'])
                        
                        if np.std(sentiments[:, 0]) > 0 and np.std(rewards) > 0:
                            correlation = np.corrcoef(sentiments[:, 0], rewards)[0, 1]
                            if not np.isnan(correlation):
                                self.training_metrics['sentiment_correlation_rewards'].append(correlation)
        
        # Update sentiment-specific training metrics
        if self.sentiment_enabled and batch_sentiment_data['vix_sentiment_values']:
            self.training_metrics['vix_sentiment_values'].extend(
                batch_sentiment_data['vix_sentiment_values']
            )
            self.training_metrics['vix_momentum_values'].extend(
                batch_sentiment_data['vix_momentum_values']
            )
            self.training_metrics['high_vix_decisions'].append(
                batch_sentiment_data['high_vix_episodes']
            )
            self.training_metrics['low_vix_decisions'].append(
                batch_sentiment_data['low_vix_episodes']
            )
        
        # Portfolio selection tracking
        self.training_metrics['portfolio_selections'].extend(
            batch_sentiment_data['portfolio_decisions']
        )
        
        # Convert trajectories to tensors
        return self._process_trajectories(trajectories)
    
    def _process_trajectories(self, trajectories: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Process collected trajectories into training format
        
        Args:
            trajectories: List of trajectory data
            
        Returns:
            Processed trajectory data as tensors
        """
        # Flatten all trajectory data
        all_states = []
        all_actions = []
        all_rewards = []
        all_log_probs = []
        all_values = []
        all_dones = []
        
        for trajectory in trajectories:
            for step in trajectory:
                all_states.append(step['state'])
                all_actions.append(step['action'])
                all_rewards.append(step['reward'])
                all_log_probs.append(step['log_prob'])
                all_values.append(step['value'])
                all_dones.append(step['done'])
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions = torch.stack(all_actions).to(self.device)
        rewards = torch.FloatTensor(all_rewards).to(self.device)
        old_log_probs = torch.stack(all_log_probs).to(self.device)
        values = torch.stack(all_values).to(self.device)
        dones = torch.tensor(all_dones, dtype=torch.bool).to(self.device)
        
        # Compute advantages and returns using GAE
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store advantage statistics
        self.training_metrics['advantages'].extend(advantages.cpu().detach().numpy())
        
        return {
            'states': states,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns,
            'old_values': values
        }
    
    def update_policy(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update policy and value networks using PPO with enhanced metrics
        
        Args:
            batch_data: Batch of trajectory data
            
        Returns:
            Dictionary of training metrics including sentiment analytics
        """
        self.policy_net.train()
        self.value_net.train()
        
        policy_losses = []
        value_losses = []
        entropy_values = []
        ratios = []
        
        # Get batch size
        batch_size = batch_data['states'].shape[0]
        
        # PPO update epochs
        for epoch in range(self.config.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(batch_size, device=self.device)
            
            # Mini-batch updates
            for start in range(0, batch_size, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, batch_size)
                mb_indices = indices[start:end]
                
                # Extract mini-batch
                mb_states = batch_data['states'][mb_indices]
                mb_actions = batch_data['actions'][mb_indices]
                mb_old_log_probs = batch_data['old_log_probs'][mb_indices]
                mb_advantages = batch_data['advantages'][mb_indices]
                mb_returns = batch_data['returns'][mb_indices]
                
                # === POLICY UPDATE ===
                
                # Get current policy outputs
                new_log_probs, entropy = self.policy_net.evaluate_actions(mb_states, mb_actions)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                ratios.extend(ratio.cpu().detach().numpy())
                
                # PPO clipped objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * mb_advantages
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                entropy_loss = -self.config.entropy_coeff * entropy.mean()
                entropy_values.extend(entropy.cpu().detach().numpy())
                
                # Total policy loss
                total_policy_loss = policy_loss + entropy_loss
                
                # Update policy
                self.policy_optimizer.zero_grad()
                total_policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                policy_losses.append(total_policy_loss.item())
                
                # === VALUE UPDATE ===
                
                # Get current value estimates
                new_values = self.value_net(mb_states)
                
                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(new_values, mb_returns)
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()
                
                value_losses.append(value_loss.item())
        
        # Store training metrics
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        
        self.training_metrics['policy_losses'].append(avg_policy_loss)
        self.training_metrics['value_losses'].append(avg_value_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'mean_advantage': batch_data['advantages'].mean().item(),
            'mean_return': batch_data['returns'].mean().item(),
            'mean_entropy': np.mean(entropy_values),
            'mean_ratio': np.mean(ratios),
            'clip_fraction': np.mean([r > 1 + self.config.clip_epsilon or r < 1 - self.config.clip_epsilon for r in ratios])
        }
    
    def train_iteration(self) -> Dict[str, float]:
        """
        Single training iteration with enhanced sentiment metrics
        
        Returns:
            Comprehensive training metrics for this iteration
        """
        # Collect trajectories
        batch_data = self.collect_trajectories(self.config.batch_size)
        
        # Update networks
        update_metrics = self.update_policy(batch_data)
        
        # Update learning rates
        self.policy_scheduler.step()
        self.value_scheduler.step()
        
        # Increment iteration counter
        self.iteration += 1
        
        # Base metrics
        metrics = {
            'iteration': self.iteration,
            'total_timesteps': self.total_timesteps,
            'total_episodes': self.total_episodes,
            'mean_episode_reward': np.mean(list(self.training_metrics['total_rewards'])),
            'mean_episode_length': np.mean(list(self.training_metrics['episode_lengths'])),
            'mean_goal_success_rate': np.mean(list(self.training_metrics['goal_success_rates'])),
            'policy_lr': self.policy_scheduler.get_last_lr()[0],
            'value_lr': self.value_scheduler.get_last_lr()[0],
            **update_metrics
        }
        
        # Portfolio selection statistics
        if self.training_metrics['portfolio_selections']:
            portfolio_counts = np.bincount(
                list(self.training_metrics['portfolio_selections']), 
                minlength=self.num_portfolios
            )
            portfolio_probs = portfolio_counts / max(portfolio_counts.sum(), 1)
            
            metrics.update({
                'portfolio_entropy': -np.sum(portfolio_probs * np.log(portfolio_probs + 1e-10)),
                'most_selected_portfolio': int(np.argmax(portfolio_counts)),
                'portfolio_concentration': np.max(portfolio_probs)
            })
        
        # Sentiment-specific metrics
        if self.sentiment_enabled:
            if self.training_metrics['vix_sentiment_values']:
                metrics.update({
                    'mean_vix_sentiment': np.mean(list(self.training_metrics['vix_sentiment_values'])),
                    'std_vix_sentiment': np.std(list(self.training_metrics['vix_sentiment_values'])),
                    'mean_vix_momentum': np.mean(list(self.training_metrics['vix_momentum_values'])),
                    'std_vix_momentum': np.std(list(self.training_metrics['vix_momentum_values']))
                })
            
            if self.training_metrics['sentiment_correlation_rewards']:
                metrics['sentiment_reward_correlation'] = np.mean(
                    list(self.training_metrics['sentiment_correlation_rewards'])
                )
            
            if self.training_metrics['high_vix_decisions']:
                metrics.update({
                    'high_vix_episodes_ratio': np.mean(list(self.training_metrics['high_vix_decisions'])) / max(self.config.batch_size, 1),
                    'low_vix_episodes_ratio': np.mean(list(self.training_metrics['low_vix_decisions'])) / max(self.config.batch_size, 1)
                })
        
        return metrics
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action for given observation
        
        Args:
            observation: Environment observation (with or without sentiment)
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action array [goal_action, portfolio_action]
        """
        self.policy_net.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            actions, _ = self.policy_net.get_action_and_log_prob(
                state_tensor,
                deterministic=deterministic
            )
            return actions.squeeze(0).cpu().detach().numpy()
    
    def get_sentiment_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis from training history
        
        Returns:
            Dictionary with sentiment insights
        """
        if not self.sentiment_enabled:
            return {'sentiment_enabled': False}
        
        analysis = {'sentiment_enabled': True}
        
        # VIX sentiment distribution
        if self.training_metrics['vix_sentiment_values']:
            sentiment_values = list(self.training_metrics['vix_sentiment_values'])
            analysis['vix_sentiment_stats'] = {
                'mean': np.mean(sentiment_values),
                'std': np.std(sentiment_values),
                'min': np.min(sentiment_values),
                'max': np.max(sentiment_values),
                'percentile_25': np.percentile(sentiment_values, 25),
                'percentile_75': np.percentile(sentiment_values, 75)
            }
        
        # VIX momentum distribution
        if self.training_metrics['vix_momentum_values']:
            momentum_values = list(self.training_metrics['vix_momentum_values'])
            analysis['vix_momentum_stats'] = {
                'mean': np.mean(momentum_values),
                'std': np.std(momentum_values),
                'min': np.min(momentum_values),
                'max': np.max(momentum_values)
            }
        
        # Sentiment-reward correlation
        if self.training_metrics['sentiment_correlation_rewards']:
            correlations = list(self.training_metrics['sentiment_correlation_rewards'])
            analysis['sentiment_reward_correlation'] = {
                'mean': np.mean(correlations),
                'std': np.std(correlations),
                'positive_correlations': np.mean([c > 0 for c in correlations])
            }
        
        return analysis
    
    def save(self, filepath: str, include_sentiment_analysis: bool = True):
        """
        Save agent state with sentiment analysis
        
        Args:
            filepath: Path to save the agent
            include_sentiment_analysis: Whether to include sentiment analysis
        """
        # Convert config to dict
        config_dict = {
            'n_traj': self.config.n_traj,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'clip_epsilon': self.config.clip_epsilon,
            'n_neurons': self.config.n_neurons,
            'ppo_epochs': self.config.ppo_epochs,
            'mini_batch_size': self.config.mini_batch_size,
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda,
            'entropy_coeff': self.config.entropy_coeff,
            'value_loss_coeff': self.config.value_loss_coeff,
            'max_grad_norm': self.config.max_grad_norm,
            'time_horizon': self.config.time_horizon,
            'device': str(self.device)
        }
        
        # Convert training metrics to serializable format
        training_metrics_dict = {}
        for key, value in self.training_metrics.items():
            training_metrics_dict[key] = list(value)
        
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config_dict': config_dict,
            'agent_config': self.agent_config,
            'training_metrics': training_metrics_dict,
            'total_timesteps': self.total_timesteps,
            'iteration': self.iteration
        }
        
        # Add sentiment analysis
        if include_sentiment_analysis:
            checkpoint['sentiment_analysis'] = self.get_sentiment_analysis()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"SentimentAwarePPOAgent saved to {filepath}")
        
        # Save human-readable configuration
        config_path = Path(filepath).parent / f"{Path(filepath).stem}_config.json"
        
        # Convert to JSON-serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        with open(config_path, 'w') as f:
            json.dump(convert_to_serializable({
                'agent_config': self.agent_config,
                'training_config': config_dict,
                'sentiment_analysis': checkpoint.get('sentiment_analysis', {})
            }), f, indent=2)
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Load network weights
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        # Load training state
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.iteration = checkpoint.get('iteration', 0)
        
        # Load training metrics if available
        if 'training_metrics' in checkpoint:
            for key, value in checkpoint['training_metrics'].items():
                if key in self.training_metrics:
                    self.training_metrics[key].extend(value)
        
        self.logger.info(f"SentimentAwarePPOAgent loaded from {filepath}")


def create_sentiment_ppo_agent(
    env,
    sentiment_enabled: bool = True,
    policy_type: str = "standard",
    encoder_type: str = "feature",
    config: TrainingConfig = None,
    device: str = None
) -> SentimentAwarePPOAgent:
    """
    Factory function to create sentiment-aware PPO agents
    
    Args:
        env: GBWM environment
        sentiment_enabled: Whether to use sentiment features
        policy_type: Type of policy network
        encoder_type: Type of state encoder
        config: Training configuration
        device: Device to use
        
    Returns:
        Initialized sentiment-aware PPO agent
    """
    return SentimentAwarePPOAgent(
        env=env,
        config=config,
        policy_type=policy_type,
        encoder_type=encoder_type,
        device=device,
        sentiment_enabled=sentiment_enabled
    )


def test_sentiment_ppo_agent():
    """Test function for sentiment-aware PPO agent"""
    print("Testing sentiment-aware PPO agent...")
    
    try:
        # Mock environment for testing
        class MockEnv:
            def __init__(self, sentiment_enabled=True):
                self.sentiment_enabled = sentiment_enabled
                self.state_dim = 4 if sentiment_enabled else 2
                
                from gymnasium import spaces
                self.observation_space = spaces.Box(
                    low=np.array([0.0] * self.state_dim),
                    high=np.array([1.0] * self.state_dim),
                    dtype=np.float32
                )
                self.action_space = spaces.MultiDiscrete([2, 15])
                
                self.time_step = 0
                self.max_steps = 5
            
            def reset(self):
                self.time_step = 0
                obs = np.random.rand(self.state_dim).astype(np.float32)
                return obs, {}
            
            def step(self, action):
                self.time_step += 1
                next_obs = np.random.rand(self.state_dim).astype(np.float32)
                reward = np.random.randn()
                done = self.time_step >= self.max_steps
                info = {
                    'goal_available': np.random.choice([True, False]),
                    'goal_taken': action[0] == 1,
                    'sentiment_features': next_obs[2:4] if self.sentiment_enabled else None
                }
                return next_obs, reward, done, False, info
        
        # Test with sentiment enabled
        env_sentiment = MockEnv(sentiment_enabled=True)
        
        # Create minimal config
        from config.training_config import TrainingConfig
        test_config = TrainingConfig(
            batch_size=2,
            ppo_epochs=1,
            mini_batch_size=2,
            time_horizon=5,
            n_neurons=32
        )
        
        agent = SentimentAwarePPOAgent(
            env=env_sentiment,
            config=test_config,
            policy_type="standard",
            encoder_type="simple",
            sentiment_enabled=True
        )
        
        print("✓ Agent initialized with sentiment support")
        
        # Test trajectory collection
        batch_data = agent.collect_trajectories(num_trajectories=2)
        
        required_keys = ['states', 'actions', 'old_log_probs', 'advantages', 'returns']
        for key in required_keys:
            assert key in batch_data, f"Missing key {key} in batch data"
        
        print(f"✓ Trajectory collection: collected {batch_data['states'].shape[0]} steps")
        
        # Test policy update
        update_metrics = agent.update_policy(batch_data)
        
        assert 'policy_loss' in update_metrics, "Missing policy loss"
        assert 'value_loss' in update_metrics, "Missing value loss"
        
        print("✓ Policy update successful")
        
        # Test prediction
        test_obs = np.random.rand(4).astype(np.float32)
        action = agent.predict(test_obs, deterministic=True)
        
        assert action.shape == (2,), f"Wrong action shape: {action.shape}"
        print(f"✓ Prediction: action={action}")
        
        # Test sentiment analysis
        sentiment_analysis = agent.get_sentiment_analysis()
        assert sentiment_analysis['sentiment_enabled'], "Sentiment should be enabled"
        print("✓ Sentiment analysis generated")
        
        # Test without sentiment
        env_no_sentiment = MockEnv(sentiment_enabled=False)
        
        agent_no_sentiment = SentimentAwarePPOAgent(
            env=env_no_sentiment,
            config=test_config,
            sentiment_enabled=False
        )
        
        print("✓ Agent initialized without sentiment support")
        
        print("All sentiment PPO agent tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"✗ Sentiment PPO agent test failed: {e}")
        return False


if __name__ == "__main__":
    test_sentiment_ppo_agent()
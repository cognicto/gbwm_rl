"""
Feature Encoders for Sentiment-Aware GBWM

This module provides specialized encoders for heterogeneous state features
in the sentiment-augmented GBWM system.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any


class FeatureEncoder(nn.Module):
    """
    Separate encoders for heterogeneous state features

    Processes different types of state information with specialized networks:
    - Time: Normalized time progression [0, 1]
    - Wealth: Normalized wealth level [0, ∞)
    - Sentiment: VIX-based market sentiment (2 or 3 features)

    Supports both 4D and 5D state inputs:
    - 4D: [time, wealth, vix_sentiment, vix_momentum]
    - 5D: [time, wealth, vix_level, vix_avg, vix_momentum] (monthly env)

    Architecture:
        Time: 1 → 16 (Tanh activation)
        Wealth: 1 → 32 (Tanh activation)
        Sentiment: 2-3 → 16 (Tanh activation)
        Fusion: 64 → 64 (Tanh activation)
    """

    def __init__(self, time_dim: int = 16, wealth_dim: int = 32, sentiment_dim: int = 16,
                 input_dim: int = 4):
        """
        Initialize feature encoder

        Args:
            time_dim: Dimensionality of time encoding
            wealth_dim: Dimensionality of wealth encoding
            sentiment_dim: Dimensionality of sentiment encoding
            input_dim: Input state dimensionality (4 or 5)
        """
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.wealth_dim = wealth_dim
        self.sentiment_dim = sentiment_dim
        self.output_dim = time_dim + wealth_dim + sentiment_dim

        # Sentiment features: 2 for 4D input, 3 for 5D input
        sentiment_features = input_dim - 2  # Subtract time and wealth

        # Individual feature encoders
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.Tanh()
        )

        self.wealth_encoder = nn.Sequential(
            nn.Linear(1, wealth_dim),
            nn.Tanh()
        )

        self.sentiment_encoder = nn.Sequential(
            nn.Linear(sentiment_features, sentiment_dim),  # 2 or 3 → sentiment_dim
            nn.Tanh()
        )

        # Fusion layer - output matches hidden_dim for consistency
        self.fusion_output_dim = 64  # Fixed output dimension
        self.fusion = nn.Sequential(
            nn.Linear(self.output_dim, self.fusion_output_dim),
            nn.Tanh()
        )

        # Initialize weights
        self._init_weights()

    def get_output_dim(self) -> int:
        """Return the output dimension of this encoder"""
        return self.fusion_output_dim

    def _init_weights(self):
        """Orthogonal initialization for stable training"""
        for module in [self.time_encoder, self.wealth_encoder,
                      self.sentiment_encoder, self.fusion]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature encoder

        Args:
            state: (batch_size, 4 or 5) tensor
                   4D: [time, wealth, vix_sentiment, vix_momentum]
                   5D: [time, wealth, vix_level, vix_avg, vix_momentum]

        Returns:
            encoded: (batch_size, 64) encoded feature representation
        """
        # Split state into components
        time = state[:, 0:1]        # (batch_size, 1)
        wealth = state[:, 1:2]      # (batch_size, 1)
        sentiment = state[:, 2:]    # (batch_size, 2 or 3) - all remaining features

        # Encode each component
        time_enc = self.time_encoder(time)          # (batch_size, time_dim)
        wealth_enc = self.wealth_encoder(wealth)    # (batch_size, wealth_dim)
        sentiment_enc = self.sentiment_encoder(sentiment)  # (batch_size, sentiment_dim)
        
        # Concatenate encoded features
        combined = torch.cat([time_enc, wealth_enc, sentiment_enc], dim=1)  # (batch_size, 64)
        
        # Apply fusion layer
        encoded = self.fusion(combined)  # (batch_size, 64)
        
        return encoded


class SimpleEncoder(nn.Module):
    """
    Simple alternative encoder for direct state processing
    
    Use when feature encoder approach is too complex or for compatibility.
    Processes the full state vector directly through dense layers.
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64):
        """
        Initialize simple encoder
        
        Args:
            input_dim: Input state dimensionality (2 for baseline, 4 for sentiment)
            hidden_dim: Hidden layer dimensionality
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization"""
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def get_output_dim(self) -> int:
        """Return the output dimension of this encoder"""
        return self.hidden_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through simple encoder
        
        Args:
            state: (batch_size, input_dim) state tensor
            
        Returns:
            encoded: (batch_size, hidden_dim) encoded representation
        """
        return self.encoder(state)


class AdaptiveEncoder(nn.Module):
    """
    Adaptive encoder that handles 2D, 4D, and 5D state inputs

    Automatically detects input dimensionality and applies appropriate encoding:
    - 2D input: [time, wealth] → uses simple encoding
    - 4D input: [time, wealth, vix_sentiment, vix_momentum] → uses feature encoding
    - 5D input: [time, wealth, vix_level, vix_avg, vix_momentum] → uses feature encoding
    """

    def __init__(self, time_dim: int = 16, wealth_dim: int = 32, sentiment_dim: int = 16):
        """
        Initialize adaptive encoder

        Args:
            time_dim: Time encoding dimensionality
            wealth_dim: Wealth encoding dimensionality
            sentiment_dim: Sentiment encoding dimensionality
        """
        super().__init__()

        # Feature encoder for 4D input (legacy)
        self.feature_encoder_4d = FeatureEncoder(time_dim, wealth_dim, sentiment_dim, input_dim=4)

        # Feature encoder for 5D input (monthly env)
        self.feature_encoder_5d = FeatureEncoder(time_dim, wealth_dim, sentiment_dim, input_dim=5)

        # Simple encoder for 2D input
        self.simple_encoder_2d = SimpleEncoder(input_dim=2, hidden_dim=64)

        # Track what mode we're in
        self.register_buffer('_input_dim', torch.tensor(-1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Adaptive forward pass

        Args:
            state: (batch_size, state_dim) where state_dim ∈ {2, 4, 5}

        Returns:
            encoded: (batch_size, 64) encoded representation
        """
        input_dim = state.size(-1)

        if input_dim == 2:
            # 2D state: [time, wealth]
            return self.simple_encoder_2d(state)
        elif input_dim == 4:
            # 4D state: [time, wealth, vix_sentiment, vix_momentum]
            return self.feature_encoder_4d(state)
        elif input_dim == 5:
            # 5D state: [time, wealth, vix_level, vix_avg, vix_momentum] (monthly)
            return self.feature_encoder_5d(state)
        else:
            raise ValueError(f"Unsupported state dimensionality: {input_dim}. Expected 2, 4, or 5.")

    def get_output_dim(self) -> int:
        """Return the output dimension of this encoder (all sub-encoders output 64)"""
        return 64


class AttentionEncoder(nn.Module):
    """
    Attention-based encoder for learning feature importance
    
    Uses self-attention to learn which state components are most relevant
    for different decision contexts. More sophisticated than fixed encoders.
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_heads: int = 4):
        """
        Initialize attention encoder
        
        Args:
            input_dim: Input state dimensionality
            hidden_dim: Hidden representation size
            num_heads: Number of attention heads
        """
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
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention
        
        Args:
            state: (batch_size, input_dim) state tensor
            
        Returns:
            encoded: (batch_size, hidden_dim) encoded representation
        """
        batch_size = state.size(0)
        
        # Project to hidden dimension
        projected = self.input_projection(state)  # (batch_size, hidden_dim)
        
        # Add sequence dimension for attention (treat features as sequence)
        # Reshape: (batch_size, hidden_dim) → (batch_size, 1, hidden_dim)
        seq_input = projected.unsqueeze(1)
        
        # Self-attention
        attn_output, _ = self.self_attention(seq_input, seq_input, seq_input)
        
        # Remove sequence dimension: (batch_size, 1, hidden_dim) → (batch_size, hidden_dim)
        attn_output = attn_output.squeeze(1)
        
        # Residual connection + layer norm
        normed_attn = self.layer_norm(projected + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(normed_attn)
        
        # Final residual + norm
        encoded = self.final_norm(normed_attn + ff_output)

        return encoded

    def get_output_dim(self) -> int:
        """Return the output dimension of this encoder"""
        return self.hidden_dim


class PerFeatureAttentionEncoder(nn.Module):
    """
    Per-Feature Attention Encoder for interpretable financial RL
    
    Processes each input feature separately and uses cross-attention to learn
    feature importance dynamically. Designed for 5D financial state:
    [time_progress, wealth_ratio, vix_level, vix_average, vix_momentum]
    
    Benefits:
    1. Interpretable feature importance via attention weights
    2. Financial domain knowledge integration  
    3. Dynamic context adaptation (market regime detection)
    4. Enhanced feature interactions
    
    Architecture:
        5D state → [token1, token2, token3, token4, token5] → cross-attention → 64D output
    """
    
    def __init__(self, input_dim: int = 5, feature_embed_dim: int = 32, num_heads: int = 4, output_dim: int = 64):
        """
        Initialize per-feature attention encoder
        
        Args:
            input_dim: Input state dimensionality (expected: 5)
            feature_embed_dim: Embedding dimension for each feature  
            num_heads: Number of attention heads
            output_dim: Output dimension (default: 64 for consistency)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_embed_dim = feature_embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        # Individual feature embeddings with financial semantics
        if input_dim == 5:
            # Monthly environment: [time, wealth, vix_level, vix_avg, vix_momentum]
            self.time_embedding = nn.Linear(1, feature_embed_dim)
            self.wealth_embedding = nn.Linear(1, feature_embed_dim) 
            self.vix_level_embedding = nn.Linear(1, feature_embed_dim)
            self.vix_avg_embedding = nn.Linear(1, feature_embed_dim)
            self.vix_momentum_embedding = nn.Linear(1, feature_embed_dim)
            self.feature_names = ['time', 'wealth', 'vix_level', 'vix_avg', 'vix_momentum']
        elif input_dim == 4:
            # Legacy environment: [time, wealth, vix_sentiment, vix_momentum]
            self.time_embedding = nn.Linear(1, feature_embed_dim)
            self.wealth_embedding = nn.Linear(1, feature_embed_dim)
            self.vix_sentiment_embedding = nn.Linear(1, feature_embed_dim) 
            self.vix_momentum_embedding = nn.Linear(1, feature_embed_dim)
            self.feature_names = ['time', 'wealth', 'vix_sentiment', 'vix_momentum']
        else:
            raise ValueError(f"PerFeatureAttentionEncoder requires 4D or 5D input, got {input_dim}D")
        
        # Cross-feature attention - allows features to attend to each other
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(feature_embed_dim)
        
        # Feed-forward network for feature refinement
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_embed_dim, feature_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_embed_dim * 2, feature_embed_dim)
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(feature_embed_dim)
        
        # Output projection to match expected dimensions
        self.output_projection = nn.Sequential(
            nn.Linear(feature_embed_dim * input_dim, output_dim),
            nn.Tanh()
        )
        
        # Store latest attention weights for interpretability
        self.register_buffer('_last_attention_weights', torch.zeros(1, input_dim, input_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with per-feature attention
        
        Args:
            state: (batch_size, input_dim) state tensor
                   5D: [time, wealth, vix_level, vix_avg, vix_momentum] 
                   4D: [time, wealth, vix_sentiment, vix_momentum]
        
        Returns:
            encoded: (batch_size, output_dim) encoded representation
        """
        batch_size = state.size(0)
        
        # Extract and embed individual features
        if self.input_dim == 5:
            # 5D monthly environment
            time, wealth, vix_level, vix_avg, vix_momentum = state.chunk(5, dim=-1)
            
            embeddings = torch.stack([
                self.time_embedding(time),
                self.wealth_embedding(wealth), 
                self.vix_level_embedding(vix_level),
                self.vix_avg_embedding(vix_avg),
                self.vix_momentum_embedding(vix_momentum)
            ], dim=1)  # (batch_size, 5, feature_embed_dim)
            
        elif self.input_dim == 4:
            # 4D legacy environment  
            time, wealth, vix_sentiment, vix_momentum = state.chunk(4, dim=-1)
            
            embeddings = torch.stack([
                self.time_embedding(time),
                self.wealth_embedding(wealth),
                self.vix_sentiment_embedding(vix_sentiment), 
                self.vix_momentum_embedding(vix_momentum)
            ], dim=1)  # (batch_size, 4, feature_embed_dim)
        
        # Cross-attention between features
        attended, attention_weights = self.cross_attention(
            embeddings, embeddings, embeddings  # Q, K, V
        )
        
        # Store attention weights for interpretability
        self._last_attention_weights = attention_weights.detach().mean(dim=0, keepdim=True)
        
        # Residual connection + layer norm
        normed_attended = self.layer_norm(embeddings + attended)
        
        # Feed-forward refinement
        ff_output = self.feed_forward(normed_attended)
        
        # Final residual + norm
        refined_features = self.final_norm(normed_attended + ff_output)
        
        # Flatten and project to output dimension
        flattened = refined_features.flatten(1)  # (batch_size, input_dim * feature_embed_dim)
        output = self.output_projection(flattened)  # (batch_size, output_dim)
        
        return output
    
    def get_output_dim(self) -> int:
        """Return the output dimension of this encoder"""
        return self.output_dim
    
    def get_attention_weights(self) -> torch.Tensor:
        """
        Get the latest attention weights for interpretability
        
        Returns:
            attention_weights: (input_dim, input_dim) attention matrix
                             attention_weights[i][j] = how much feature i attends to feature j
        """
        return self._last_attention_weights.squeeze(0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores based on attention weights
        
        Returns:
            Dictionary mapping feature names to importance scores (0-1)
        """
        if self._last_attention_weights.sum() == 0:
            # No forward pass yet, return uniform weights
            uniform_weight = 1.0 / self.input_dim
            return {name: uniform_weight for name in self.feature_names}
        
        # Average attention received by each feature (column-wise mean)
        attention_matrix = self._last_attention_weights.squeeze(0)  # (input_dim, input_dim)
        importance_scores = attention_matrix.mean(dim=0)  # (input_dim,)
        
        # Normalize to sum to 1
        importance_scores = importance_scores / importance_scores.sum()
        
        return {
            name: score.item() 
            for name, score in zip(self.feature_names, importance_scores)
        }
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive attention analysis for financial interpretation
        
        Returns:
            Dictionary with attention patterns, feature importance, and regime indicators
        """
        attention_matrix = self.get_attention_weights()  # (input_dim, input_dim) 
        feature_importance = self.get_feature_importance()
        
        summary = {
            'feature_importance': feature_importance,
            'attention_matrix': attention_matrix.cpu().numpy(),
            'feature_names': self.feature_names
        }
        
        # Financial regime analysis for 5D input
        if self.input_dim == 5:
            vix_attention = (
                feature_importance.get('vix_level', 0) + 
                feature_importance.get('vix_avg', 0) + 
                feature_importance.get('vix_momentum', 0)
            )
            
            # Market regime classification based on attention patterns
            if vix_attention > 0.6:
                regime = "crisis"  # High VIX attention = market stress
            elif feature_importance.get('vix_momentum', 0) > 0.3:
                regime = "recovery"  # VIX momentum focus = transitioning
            elif feature_importance.get('time', 0) > 0.4:
                regime = "deadline_driven"  # Time pressure dominates
            else:
                regime = "calm"  # Balanced attention = normal conditions
                
            summary.update({
                'vix_total_attention': vix_attention,
                'predicted_regime': regime,
                'time_vs_wealth_ratio': (
                    feature_importance.get('time', 0) / 
                    max(feature_importance.get('wealth', 0), 1e-8)
                )
            })
        
        return summary


def create_encoder(
    encoder_type: str = "feature",
    input_dim: int = 4,
    hidden_dim: int = 64,
    **kwargs
) -> nn.Module:
    """
    Factory function to create encoders

    Args:
        encoder_type: Type of encoder ('feature', 'simple', 'adaptive', 'attention', 'per_feature_attention')
        input_dim: Input dimensionality (2 for pure RL, 4 for legacy sentiment, 5 for monthly)
        hidden_dim: Hidden layer dimensionality
        **kwargs: Additional encoder parameters

    Returns:
        Initialized encoder module
    """
    if encoder_type == "feature":
        if input_dim not in (4, 5):
            raise ValueError(f"FeatureEncoder requires 4D or 5D input, got {input_dim}D")
        # FeatureEncoder doesn't use hidden_dim, it has fixed output of 64
        # Remove hidden_dim from kwargs if present
        feature_kwargs = {k: v for k, v in kwargs.items() if k != 'hidden_dim'}
        return FeatureEncoder(input_dim=input_dim, **feature_kwargs)

    elif encoder_type == "simple":
        return SimpleEncoder(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)

    elif encoder_type == "adaptive":
        return AdaptiveEncoder(**kwargs)

    elif encoder_type == "attention":
        return AttentionEncoder(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)

    elif encoder_type == "per_feature_attention":
        if input_dim not in (4, 5):
            raise ValueError(f"PerFeatureAttentionEncoder requires 4D or 5D input, got {input_dim}D")
        # PerFeatureAttentionEncoder uses output_dim instead of hidden_dim
        per_feature_kwargs = {k: v for k, v in kwargs.items() if k != 'hidden_dim'}
        if 'output_dim' not in per_feature_kwargs:
            per_feature_kwargs['output_dim'] = hidden_dim
        return PerFeatureAttentionEncoder(input_dim=input_dim, **per_feature_kwargs)

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def test_encoders():
    """Test function for all encoder types"""
    print("Testing feature encoders...")
    
    # Test data
    batch_size = 32
    state_2d = torch.randn(batch_size, 2)  # [time, wealth]
    state_4d = torch.randn(batch_size, 4)  # [time, wealth, vix_sentiment, vix_momentum]
    state_5d = torch.randn(batch_size, 5)  # [time, wealth, vix_level, vix_avg, vix_momentum]
    
    try:
        # Test FeatureEncoder
        feature_encoder = FeatureEncoder()
        output = feature_encoder(state_4d)
        assert output.shape == (batch_size, 64), f"Wrong output shape: {output.shape}"
        print("✓ FeatureEncoder test passed")
        
        # Test SimpleEncoder
        simple_encoder_2d = SimpleEncoder(input_dim=2)
        output = simple_encoder_2d(state_2d)
        assert output.shape == (batch_size, 64), f"Wrong output shape: {output.shape}"
        print("✓ SimpleEncoder (2D) test passed")
        
        simple_encoder_4d = SimpleEncoder(input_dim=4)
        output = simple_encoder_4d(state_4d)
        assert output.shape == (batch_size, 64), f"Wrong output shape: {output.shape}"
        print("✓ SimpleEncoder (4D) test passed")
        
        # Test AdaptiveEncoder
        adaptive_encoder = AdaptiveEncoder()
        output_2d = adaptive_encoder(state_2d)
        output_4d = adaptive_encoder(state_4d)
        output_5d = adaptive_encoder(state_5d)
        assert output_2d.shape == (batch_size, 64), f"Wrong 2D output shape: {output_2d.shape}"
        assert output_4d.shape == (batch_size, 64), f"Wrong 4D output shape: {output_4d.shape}"
        assert output_5d.shape == (batch_size, 64), f"Wrong 5D output shape: {output_5d.shape}"
        print("✓ AdaptiveEncoder test passed")
        
        # Test AttentionEncoder
        attention_encoder = AttentionEncoder(input_dim=4)
        output = attention_encoder(state_4d)
        assert output.shape == (batch_size, 64), f"Wrong output shape: {output.shape}"
        print("✓ AttentionEncoder test passed")
        
        # Test PerFeatureAttentionEncoder
        per_feature_encoder_5d = PerFeatureAttentionEncoder(input_dim=5)
        output = per_feature_encoder_5d(state_5d)
        assert output.shape == (batch_size, 64), f"Wrong output shape: {output.shape}"
        
        # Test attention weights and interpretability
        attention_weights = per_feature_encoder_5d.get_attention_weights()
        assert attention_weights.shape == (5, 5), f"Wrong attention weights shape: {attention_weights.shape}"
        
        feature_importance = per_feature_encoder_5d.get_feature_importance()
        expected_features = ['time', 'wealth', 'vix_level', 'vix_avg', 'vix_momentum']
        assert all(feat in feature_importance for feat in expected_features), "Missing feature in importance dict"
        
        attention_summary = per_feature_encoder_5d.get_attention_summary()
        assert 'predicted_regime' in attention_summary, "Missing regime prediction"
        assert 'vix_total_attention' in attention_summary, "Missing VIX attention analysis"
        
        print("✓ PerFeatureAttentionEncoder (5D) test passed")
        
        # Test 4D PerFeatureAttentionEncoder
        per_feature_encoder_4d = PerFeatureAttentionEncoder(input_dim=4)
        output_4d = per_feature_encoder_4d(state_4d)
        assert output_4d.shape == (batch_size, 64), f"Wrong 4D output shape: {output_4d.shape}"
        print("✓ PerFeatureAttentionEncoder (4D) test passed")
        
        # Test factory function
        encoders = [
            create_encoder("feature", input_dim=4),
            create_encoder("simple", input_dim=2),
            create_encoder("adaptive"),
            create_encoder("attention", input_dim=4),
            create_encoder("per_feature_attention", input_dim=5),
            create_encoder("per_feature_attention", input_dim=4)
        ]
        print("✓ Encoder factory test passed")
        
        # Test interpretability features
        print("\nTesting interpretability features:")
        
        # Run forward pass to generate attention
        _ = per_feature_encoder_5d(state_5d)
        
        importance = per_feature_encoder_5d.get_feature_importance()
        print(f"Feature importance: {importance}")
        
        summary = per_feature_encoder_5d.get_attention_summary()
        print(f"Predicted regime: {summary['predicted_regime']}")
        print(f"VIX total attention: {summary['vix_total_attention']:.3f}")
        
        print("✓ Interpretability features test passed")
        
        print("All encoder tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"✗ Encoder test failed: {e}")
        return False


if __name__ == "__main__":
    test_encoders()
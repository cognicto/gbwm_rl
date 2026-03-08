#!/usr/bin/env python3
"""
Demonstration of Per-Feature Attention in Financial RL

This script demonstrates how the per-feature attention encoder works
and shows its interpretability features for financial decision making.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import torch
    import numpy as np
    from src.models.feature_encoders import PerFeatureAttentionEncoder
    from src.models.sentiment_policy_network import SentimentAwarePolicyNetwork
except ImportError as e:
    print(f"Import error: {e}")
    print("This demo requires PyTorch. Please install it to run the demonstration.")
    sys.exit(1)


def create_market_scenarios():
    """Create different market scenarios to demonstrate attention patterns"""
    
    scenarios = {
        'calm_market': torch.tensor([
            [0.3, 0.6, 15.0, 18.5, -0.2],  # Low VIX, positive momentum
            [0.7, 0.8, 12.5, 16.2, -0.5],  # Very low VIX, continuing decline
        ]),
        
        'crisis_market': torch.tensor([
            [0.1, 0.2, 45.0, 35.2, 8.5],   # High VIX, rising fast
            [0.5, 0.4, 55.0, 42.1, 12.0],  # Very high VIX, crisis mode
        ]),
        
        'recovery_market': torch.tensor([
            [0.6, 0.5, 25.0, 32.5, -5.0],  # Moderate VIX, declining momentum
            [0.8, 0.7, 20.0, 28.1, -7.5],  # VIX normalizing, strong decline
        ]),
        
        'deadline_pressure': torch.tensor([
            [0.95, 0.3, 18.0, 20.5, 1.0],  # Near deadline, low wealth, moderate VIX
            [0.98, 0.1, 22.0, 19.8, 2.5],  # Very near deadline, very low wealth
        ])
    }
    
    return scenarios


def analyze_attention_patterns():
    """Analyze how attention patterns change across different market scenarios"""
    
    print("🔍 Analyzing Per-Feature Attention Patterns in Financial RL")
    print("=" * 60)
    
    # Create per-feature attention policy
    policy = SentimentAwarePolicyNetwork(
        state_dim=5,
        encoder_type="per_feature_attention",
        num_portfolios=15
    )
    
    print(f"✓ Created policy with per-feature attention encoder")
    print(f"  - Input dimensions: {policy.state_dim}")
    print(f"  - Encoder type: {policy.encoder_type}")
    print(f"  - Has attention: {policy.has_attention_encoder()}")
    print()
    
    # Get market scenarios
    scenarios = create_market_scenarios()
    
    print(" Market Scenario Analysis:")
    print("Features: [time_progress, wealth_ratio, vix_level, vix_avg, vix_momentum]")
    print()
    
    for scenario_name, states in scenarios.items():
        print(f" {scenario_name.upper().replace('_', ' ')}")
        print("-" * 40)
        
        # Run policy to generate attention
        goal_probs, portfolio_probs = policy(states)
        
        # Get attention analysis
        decision_analysis = policy.analyze_decision(states, include_attention=True)
        
        if 'attention' in decision_analysis:
            attention_summary = decision_analysis['attention']
            
            print(f"Predicted Regime: {attention_summary['predicted_regime']}")
            print(f"VIX Total Attention: {attention_summary['vix_total_attention']:.3f}")
            print(f"Time vs Wealth Ratio: {attention_summary['time_vs_wealth_ratio']:.3f}")
            print()
            
            # Show feature importance
            feature_importance = attention_summary['feature_importance']
            print("Feature Importance:")
            for feature, importance in feature_importance.items():
                bar_length = int(importance * 20)  # Scale to 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {feature:12} │{bar}│ {importance:.3f}")
            print()
            
            # Show attention matrix
            attention_matrix = attention_summary['attention_matrix']
            feature_names = attention_summary['feature_names']
            
            print("Attention Matrix (how much each feature attends to others):")
            print("     " + "".join(f"{name[:4]:>8}" for name in feature_names))
            for i, row_name in enumerate(feature_names):
                row_str = f"{row_name[:4]:>4} "
                for j in range(len(feature_names)):
                    val = attention_matrix[i, j]
                    row_str += f"{val:7.3f} "
                print(row_str)
            print()
        
        # Show action probabilities
        action_probs = decision_analysis['action_probs']
        avg_take_prob = action_probs['goal_take_prob'].mean().item()
        most_likely_portfolio = action_probs['most_likely_portfolio'][0].item()
        
        print(f"Decision Analysis:")
        print(f"  Average Take Goal Probability: {avg_take_prob:.3f}")
        print(f"  Most Likely Portfolio: {most_likely_portfolio}")
        print()
        print("=" * 60)
        print()


def demonstrate_regime_detection():
    """Demonstrate automatic market regime detection"""
    
    print(" Market Regime Detection Demo")
    print("=" * 40)
    
    # Create encoder for direct testing
    encoder = PerFeatureAttentionEncoder(input_dim=5, feature_embed_dim=32, num_heads=4)
    
    # Test different market conditions
    test_cases = [
        {
            'name': 'Normal Market',
            'state': torch.tensor([[0.5, 0.7, 16.0, 18.0, 0.5]]),
            'description': 'Moderate time, good wealth, low VIX'
        },
        {
            'name': 'Market Crisis', 
            'state': torch.tensor([[0.3, 0.4, 50.0, 45.0, 15.0]]),
            'description': 'Early time, moderate wealth, very high VIX'
        },
        {
            'name': 'Recovery Phase',
            'state': torch.tensor([[0.7, 0.6, 28.0, 35.0, -8.0]]),
            'description': 'Late time, good wealth, declining VIX'
        },
        {
            'name': 'Deadline Crisis',
            'state': torch.tensor([[0.95, 0.2, 20.0, 22.0, 2.0]]),
            'description': 'Near deadline, low wealth, moderate VIX'
        }
    ]
    
    for test_case in test_cases:
        # Forward pass to compute attention
        output = encoder(test_case['state'])
        
        # Get attention summary
        summary = encoder.get_attention_summary()
        
        print(f" {test_case['name']}")
        print(f"   {test_case['description']}")
        print(f"   State: {test_case['state'].numpy().flatten()}")
        print(f"   Predicted Regime: {summary['predicted_regime']}")
        print(f"   VIX Attention: {summary['vix_total_attention']:.3f}")
        
        if 'time_vs_wealth_ratio' in summary:
            print(f"   Time/Wealth Ratio: {summary['time_vs_wealth_ratio']:.3f}")
        
        print()


def performance_comparison():
    """Compare parameter counts between encoders"""
    
    print("⚡ Performance Comparison")
    print("=" * 30)
    
    # Create different encoders
    encoders = {
        'Simple (5D)': ('simple', 5),
        'Feature (5D)': ('feature', 5),  
        'Attention (5D)': ('attention', 5),
        'Per-Feature (5D)': ('per_feature_attention', 5),
    }
    
    print("Encoder Type           Parameters    Output Dim")
    print("-" * 45)
    
    for name, (encoder_type, input_dim) in encoders.items():
        try:
            if encoder_type == 'per_feature_attention':
                from src.models.feature_encoders import PerFeatureAttentionEncoder
                encoder = PerFeatureAttentionEncoder(input_dim=input_dim)
            else:
                from src.models.feature_encoders import create_encoder
                encoder = create_encoder(encoder_type, input_dim=input_dim, hidden_dim=64)
            
            # Count parameters
            total_params = sum(p.numel() for p in encoder.parameters())
            output_dim = encoder.get_output_dim()
            
            print(f"{name:<20} {total_params:>8,}     {output_dim:>6}")
            
        except Exception as e:
            print(f"{name:<20} {'Error':<8}     {'N/A':>6}")
    
    print()
    print("Note: Per-feature attention has ~6× more parameters but provides")
    print("      interpretability and dynamic feature importance weighting.")


def main():
    """Run the complete demonstration"""
    
    print(" Per-Feature Attention for Financial RL Demo")
    print("=" * 50)
    print()
    
    # Check if we have PyTorch available
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    try:
        # Run demonstrations
        analyze_attention_patterns()
        demonstrate_regime_detection() 
        performance_comparison()
        
        print(" Demo completed successfully!")
        print()
        print("Key Benefits Demonstrated:")
        print("•  Automatic market regime detection")
        print("•  Interpretable feature importance")
        print("•  Dynamic attention patterns")
        print("•  Financial domain knowledge integration")
        print()
        print("Usage in your code:")
        print("```python")
        print("policy = SentimentAwarePolicyNetwork(")
        print("    state_dim=5,")
        print("    encoder_type='per_feature_attention'")
        print(")")
        print("```")
        
    except Exception as e:
        print(f" Error during demonstration: {e}")
        print("This may be due to missing dependencies or installation issues.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
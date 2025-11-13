"""Utilities for saving and loading trained models."""
import torch
from typing import Tuple
from .dqn import MultiHeadDQNAgent


def load_trained_agent(model_path: str = "models/dqn_portfolio_agent.pt") -> Tuple[MultiHeadDQNAgent, dict]:
    """
    Load a trained DQN agent from a checkpoint.
    
    Args:
        model_path: Path to the saved model checkpoint
        
    Returns:
        Tuple of (agent, metadata) where metadata contains training info
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Recreate agent with saved parameters
    agent = MultiHeadDQNAgent(
        state_shape=checkpoint['state_shape'],
        n_stocks=checkpoint['n_stocks'],
        size_bins=checkpoint.get('size_bins', 5),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # Load trained weights
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Set to evaluation mode (greedy policy)
    agent.policy_net.eval()
    agent.target_net.eval()
    agent.eps_end = 0.0
    agent.step_count = agent.eps_decay_steps
    
    metadata = {
        'symbols': checkpoint.get('symbols', []),
        'train_results': checkpoint.get('train_results', []),
        'val_result': checkpoint.get('val_result', {}),
    }
    
    print(f"Loaded model from: {model_path}")
    print(f"Trained on {checkpoint['n_stocks']} stocks: {metadata['symbols']}")
    if metadata['val_result']:
        print(f"Validation return: {metadata['val_result'].get('pct_return', 0):.2f}%")
    
    return agent, metadata

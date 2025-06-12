"""Abstract interfaces for reward functions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class RewardFunction(ABC):
    """Abstract base class for reward functions in Pokemon Pinball environment."""
    
    @abstractmethod
    def calculate_reward(self, current_fitness: float, previous_fitness: float,
                        game_wrapper: Any, frames_played: int,
                        prev_state: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate reward based on current game state.
        
        Args:
            current_fitness: Current game score
            previous_fitness: Previous game score
            game_wrapper: PyBoy game wrapper instance
            frames_played: Number of frames played in episode
            prev_state: Previous state dictionary for tracking changes
            
        Returns:
            Calculated reward value (can be positive, negative, or zero)
        """
        pass
    
    def get_initial_state(self) -> Dict[str, Any]:
        """
        Get initial state dictionary for tracking.
        Override this if your reward function needs state tracking.
        
        Returns:
            Initial state dictionary
        """
        return {}
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get additional info about this reward function for logging.
        
        Returns:
            Dictionary with reward function metadata
        """
        return {
            'reward_function': self.__class__.__name__,
            'description': self.__doc__ or 'No description available'
        }
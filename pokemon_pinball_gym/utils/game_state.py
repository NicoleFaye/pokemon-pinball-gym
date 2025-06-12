"""Game state tracking utilities."""

from typing import Dict, Any


class GameStateTracker:
    """Tracks game state changes for reward calculation."""
    
    def __init__(self):
        """Initialize state tracker."""
        self.reset()
    
    def reset(self):
        """Reset all tracking variables."""
        self.prev_caught = 0
        self.prev_evolutions = 0
        self.prev_stages_completed = 0
        self.prev_ball_upgrades = 0
        self.prev_balls_left = 2
        self.prev_balls_lost_during_saver = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for reward functions."""
        return {
            'prev_caught': self.prev_caught,
            'prev_evolutions': self.prev_evolutions,
            'prev_stages_completed': self.prev_stages_completed,
            'prev_ball_upgrades': self.prev_ball_upgrades,
            'prev_ball_lost_during_saver': self.prev_balls_lost_during_saver,
        }
    
    def update(self, game_wrapper):
        """Update tracking state based on current game wrapper state."""
        self.prev_caught = game_wrapper.pokemon_caught_in_session
        self.prev_evolutions = game_wrapper.evolution_success_count
        
        self.prev_stages_completed = (
            game_wrapper.diglett_stages_completed +
            game_wrapper.gengar_stages_completed +
            game_wrapper.meowth_stages_completed +
            game_wrapper.seel_stages_completed +
            game_wrapper.mewtwo_stages_completed
        )
        
        self.prev_ball_upgrades = (
            game_wrapper.great_ball_upgrades +
            game_wrapper.ultra_ball_upgrades +
            game_wrapper.master_ball_upgrades
        )
        
        self.prev_balls_left = game_wrapper.balls_left
        self.prev_balls_lost_during_saver = game_wrapper.lost_ball_during_saver
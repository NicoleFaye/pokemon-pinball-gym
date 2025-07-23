"""Built-in reward function implementations."""

from typing import Any, Dict, Optional
import numpy as np
from pyboy.plugins.game_wrapper_pokemon_pinball import BallType

from .interface import RewardFunction


class BasicReward(RewardFunction):
    """Basic reward function that returns scaled score difference."""
    
    def __init__(self, scale_factor: float = 0.01):
        """
        Initialize BasicReward.
        
        Args:
            scale_factor: Multiplier for score difference
        """
        self.scale_factor = scale_factor
    
    def calculate_reward(self, current_fitness: float, previous_fitness: float,
                        game_wrapper: Any, frames_played: int,
                        prev_state: Optional[Dict[str, Any]] = None) -> float:
        """Calculate basic reward as scaled score difference."""
        return (current_fitness - previous_fitness) * self.scale_factor
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'reward_function': 'BasicReward',
            'description': 'Simple score-based reward with scaling',
            'scale_factor': self.scale_factor
        }


class CatchFocusedReward(RewardFunction):
    """Reward function that heavily emphasizes catching Pokemon."""
    
    def __init__(self, score_scale: float = 0.005, catch_bonus: float = 3.0):
        """
        Initialize CatchFocusedReward.
        
        Args:
            score_scale: Multiplier for score difference
            catch_bonus: Bonus reward for catching Pokemon
        """
        self.score_scale = score_scale
        self.catch_bonus = catch_bonus
    
    def calculate_reward(self, current_fitness: float, previous_fitness: float,
                        game_wrapper: Any, frames_played: int,
                        prev_state: Optional[Dict[str, Any]] = None) -> float:
        """Calculate reward with emphasis on Pokemon catching."""
        if prev_state is None:
            prev_state = self.get_initial_state()
            
        score_reward = (current_fitness - previous_fitness) * self.score_scale
        
        catch_reward = 0
        if game_wrapper.pokemon_caught_in_session > prev_state.get('prev_caught', 0):
            catch_reward = self.catch_bonus
            
        return score_reward + catch_reward
    
    def get_initial_state(self) -> Dict[str, Any]:
        return {'prev_caught': 0}
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'reward_function': 'CatchFocusedReward',
            'description': 'Emphasizes catching Pokemon over pure scoring',
            'score_scale': self.score_scale,
            'catch_bonus': self.catch_bonus
        }


class ComprehensiveReward(RewardFunction):
    """Comprehensive reward function with multiple objectives,
       plus a time‑survival bonus to lengthen episodes."""
    
    def __init__(self,
                 score_scale: float = 0.3,
                 catch_bonus: float = 5.0,
                 evolution_bonus: float = 10.0,
                 stage_bonus: float = 8.0,
                 ball_upgrade_bonus: float = 3.0,
                 saver_penalty: float = -3.0,
                 time_step_bonus: float = 0.001,
                 max_bonus_frames: int = 10000):
        """
        Args:
            time_step_bonus: small reward per frame to encourage survival
            max_bonus_frames: only give time_step_bonus while frames_played <= this
        """
        self.score_scale = score_scale
        self.catch_bonus = catch_bonus
        self.evolution_bonus = evolution_bonus
        self.stage_bonus = stage_bonus
        self.ball_upgrade_bonus = ball_upgrade_bonus
        self.saver_penalty = saver_penalty
        
        # new attributes
        self.time_step_bonus = time_step_bonus
        self.max_bonus_frames = max_bonus_frames
    
    def calculate_reward(self,
                         current_fitness: float,
                         previous_fitness: float,
                         game_wrapper: Any,
                         frames_played: int,
                         prev_state: Optional[Dict[str, Any]] = None) -> float:
        if prev_state is None:
            prev_state = self.get_initial_state()
        
        # 1) Base score reward (log‐scaled)
        score_diff = current_fitness - previous_fitness
        score_reward = self.score_scale * np.log(1 + score_diff / 50) if score_diff > 0 else 0
        
        additional_reward = 0
        
        # 2) Catch, evolution, stage, ball‑upgrade, saver loss (unchanged)…
        if game_wrapper.pokemon_caught_in_session > prev_state['prev_caught']:
            additional_reward += self.catch_bonus
        if game_wrapper.evolution_success_count > prev_state['prev_evolutions']:
            additional_reward += self.evolution_bonus
        total_stages = (
            game_wrapper.diglett_stages_completed +
            game_wrapper.gengar_stages_completed +
            game_wrapper.meowth_stages_completed +
            game_wrapper.seel_stages_completed +
            game_wrapper.mewtwo_stages_completed
        )
        if total_stages > prev_state['prev_stages_completed']:
            additional_reward += self.stage_bonus
        
        ball_upgrades = (
            game_wrapper.great_ball_upgrades +
            game_wrapper.ultra_ball_upgrades +
            game_wrapper.master_ball_upgrades
        )
        if ball_upgrades > prev_state['prev_ball_upgrades']:
            # scale by ball type
            if game_wrapper.ball_type == BallType.GREATBALL:
                additional_reward += self.ball_upgrade_bonus
            elif game_wrapper.ball_type == BallType.ULTRABALL:
                additional_reward += 2 * self.ball_upgrade_bonus
            elif game_wrapper.ball_type == BallType.MASTERBALL:
                additional_reward += 3 * self.ball_upgrade_bonus
            else:
                additional_reward += self.ball_upgrade_bonus
        
        saver_penalty = (
            self.saver_penalty
            if game_wrapper.lost_ball_during_saver > prev_state['prev_ball_lost_during_saver']
            else 0
        )
        
        # 3) small per-frame survival bonus up to a limit
        time_bonus = self.time_step_bonus if frames_played <= self.max_bonus_frames else 0
        
        return score_reward + additional_reward + saver_penalty + time_bonus
    
    def get_initial_state(self) -> Dict[str, Any]:
        return {
            'prev_caught': 0,
            'prev_evolutions': 0,
            'prev_stages_completed': 0,
            'prev_ball_upgrades': 0,
            'prev_ball_lost_during_saver': 0,
        }
    
    def get_info(self) -> Dict[str, Any]:
        info = {
            'reward_function': 'ComprehensiveReward',
            'description': 'Multi-objective reward balancing score, catches, evolutions, stages, and survival',
            'score_scale': self.score_scale,
            'catch_bonus': self.catch_bonus,
            'evolution_bonus': self.evolution_bonus,
            'stage_bonus': self.stage_bonus,
            'ball_upgrade_bonus': self.ball_upgrade_bonus,
            'saver_penalty': self.saver_penalty,
            'time_step_bonus': self.time_step_bonus,
            'max_bonus_frames': self.max_bonus_frames,
        }
        return info
"""Info building utilities."""

from typing import Dict, Any


class InfoBuilder:
    """Builds info dictionary for environment feedback."""
    
    @staticmethod
    def build_info(game_wrapper, fitness, frames_played, episodes_completed, episode_count, 
                   episode_mode, reset_condition, episode_complete=False, high_score=False) -> Dict[str, Any]:
        """
        Build comprehensive info dictionary.
        
        Args:
            game_wrapper: PyBoy game wrapper instance
            fitness: Current fitness/score
            frames_played: Number of frames played in episode
            episodes_completed: Total episodes completed
            episode_count: Current episode number
            episode_mode: Episode termination mode
            reset_condition: Reset condition mode
            episode_complete: Whether episode just completed
            high_score: Whether a new high score was achieved
            
        Returns:
            Dictionary containing environment info
        """
        # Basic info
        info = {
            'score': [float(game_wrapper.score)],
            'episode_return': [float(fitness)],
            'episode_length': [float(frames_played)],
            'agent_episodes_completed': [float(episodes_completed)],
            'episode_id': [float(episode_count)],
            'episode_complete': [episode_complete],
        }
        
        # Game progress info
        info.update({
            'pokemon_caught': [float(game_wrapper.pokemon_caught_in_session)],
            'evolutions': [float(game_wrapper.evolution_success_count)],
            'ball_saver_active': [float(game_wrapper.ball_saver_seconds_left > 0)],
            'current_stage': [str(game_wrapper.current_stage)],
            'ball_type': [str(game_wrapper.ball_type)],
            'special_mode_active': [float(game_wrapper.special_mode_active)],
            'pikachu_saver_charge': [float(game_wrapper.pikachu_saver_charge)]
        })
        
        # Stage completion info
        total_stages = (
            game_wrapper.diglett_stages_completed +
            game_wrapper.gengar_stages_completed +
            game_wrapper.meowth_stages_completed +
            game_wrapper.seel_stages_completed +
            game_wrapper.mewtwo_stages_completed
        )
        
        info.update({
            'diglett_stages': [float(game_wrapper.diglett_stages_completed)],
            'gengar_stages': [float(game_wrapper.gengar_stages_completed)],
            'meowth_stages': [float(game_wrapper.meowth_stages_completed)],
            'seel_stages': [float(game_wrapper.seel_stages_completed)],
            'mewtwo_stages': [float(game_wrapper.mewtwo_stages_completed)],
            'total_stages_completed': [float(total_stages)]
        })
        
        # Ball upgrade info
        total_upgrades = (
            game_wrapper.great_ball_upgrades +
            game_wrapper.ultra_ball_upgrades +
            game_wrapper.master_ball_upgrades
        )
        
        info.update({
            'great_ball_upgrades': [float(game_wrapper.great_ball_upgrades)],
            'ultra_ball_upgrades': [float(game_wrapper.ultra_ball_upgrades)],
            'master_ball_upgrades': [float(game_wrapper.master_ball_upgrades)],
            'total_ball_upgrades': [float(total_upgrades)]
        })
        
        # Ball position and velocity
        info.update({
            'ball_x': [float(game_wrapper.ball_x)],
            'ball_y': [float(game_wrapper.ball_y)],
            'ball_x_velocity': [float(game_wrapper.ball_x_velocity)],
            'ball_y_velocity': [float(game_wrapper.ball_y_velocity)]
        })
        
        # Episode configuration
        info.update({
            'episode_mode': [episode_mode],
            'reset_condition': [reset_condition],
            'balls_left': [float(game_wrapper.balls_left)]
        })
        
        if high_score:
            info['high_score'] = [True]
            
        return info
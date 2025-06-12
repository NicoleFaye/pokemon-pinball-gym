from pyboy.plugins.game_wrapper_pokemon_pinball import Stage, BallType, SpecialMode, Maps, Pokemon
"""
Reward shaping functions for Pokemon Pinball environment.
"""

class RewardShaping:
    """
    Collection of reward shaping functions for Pokemon Pinball.
    These methods operate on the environment instance to ensure proper per-instance state tracking.
    """
    
    @staticmethod
    def basic(current_fitness, previous_fitness, game_wrapper, frames_played=0, prev_state=None):
        """Basic reward shaping based on score difference."""
        # Scale down raw score difference
        return (current_fitness - previous_fitness) * 0.01 
    
    @staticmethod
    def catch_focused(current_fitness, previous_fitness, game_wrapper, frames_played=0, prev_state=None):
        """
        Reward focused on catching Pokemon.
        
        Args:
            current_fitness: Current score
            previous_fitness: Previous score
            game_wrapper: Game wrapper instance
            frames_played: Number of frames played
            prev_state: Dictionary containing previous state values for tracking
        """
        if prev_state is None:
            prev_state = {'prev_caught': 0}
            
        score_reward = (current_fitness - previous_fitness) * 0.005  # Reduced from 0.5
        
        catch_reward = 0
        if game_wrapper.pokemon_caught_in_session > prev_state.get('prev_caught', 0):
            catch_reward = 3.0  
            
        return score_reward + catch_reward
    
    @staticmethod
    def comprehensive(current_fitness, previous_fitness, game_wrapper, frames_played=0, prev_state=None):
        """
        Improved comprehensive reward function with better reward scaling.
        
        Args:
            current_fitness: Current score
            previous_fitness: Previous score
            game_wrapper: Game wrapper instance
            frames_played: Number of frames played
            prev_state: Dictionary containing previous state values for tracking
        """
        # If no state is passed, create empty state
        if prev_state is None:
            prev_state = {
                'prev_caught': 0,
                'prev_evolutions': 0, 
                'prev_stages_completed': 0,
                'prev_ball_upgrades': 0,
                'prev_ball_lost_during_saver': 0,
            }
            
        # Base score reward - logarithmic scaling to prevent score domination
        score_diff = current_fitness - previous_fitness
        if score_diff > 0:
            import numpy as np
            # More significant log scaling to better normalize large score jumps
            score_reward = 0.3 * np.log(1 + score_diff / 50)
        else:
            score_reward = 0

        # Initialize additional rewards
        additional_reward = 0
        saver_penalty = 0
        reward_components = {}

        # Pokemon catch reward - fundamental goal
        prev_caught = prev_state.get('prev_caught', 0)
        if game_wrapper.pokemon_caught_in_session > prev_caught:
            pokemon_reward = 5.0  # Increased to emphasize catching importance
            additional_reward += pokemon_reward
            reward_components["pokemon_catch"] = pokemon_reward

        # Evolution rewards - rare achievement deserves high reward
        prev_evolutions = prev_state.get('prev_evolutions', 0)
        if game_wrapper.evolution_success_count > prev_evolutions:
            evolution_reward = 10.0  # Higher reward as evolutions are major achievements
            additional_reward += evolution_reward
            reward_components["evolution"] = evolution_reward

        # Stage completion rewards - moderately difficult achievements
        total_stages_completed = (
            game_wrapper.diglett_stages_completed +
            game_wrapper.gengar_stages_completed +
            game_wrapper.meowth_stages_completed +
            game_wrapper.seel_stages_completed +
            game_wrapper.mewtwo_stages_completed
        )
        prev_stages_completed = prev_state.get('prev_stages_completed', 0)
        if total_stages_completed > prev_stages_completed:
            stage_reward = 8.0  # Higher reward for stage completion
            additional_reward += stage_reward
            reward_components["stage_completion"] = stage_reward

        # Ball upgrade rewards - stepping stones to better performance
        ball_upgrades = (
            game_wrapper.great_ball_upgrades +
            game_wrapper.ultra_ball_upgrades +
            game_wrapper.master_ball_upgrades
        )
        prev_ball_upgrades = prev_state.get('prev_ball_upgrades', 0)
        if ball_upgrades > prev_ball_upgrades:
            # Scale reward based on ball type
            if game_wrapper.ball_type == BallType.GREATBALL:
                upgrade_reward = 3.0
            elif game_wrapper.ball_type == BallType.ULTRABALL:
                upgrade_reward = 6.0
            elif game_wrapper.ball_type == BallType.MASTERBALL:
                upgrade_reward = 9.0
            else:
                upgrade_reward = 3.0
            
            additional_reward += upgrade_reward
            reward_components["ball_upgrade"] = upgrade_reward

        # Penalty for losing ball - teaches ball preservation
        if game_wrapper.lost_ball_during_saver > prev_state.get('prev_ball_lost_during_saver', 0):
            saver_penalty = -3.0  # Reduced from -5.0 to not overly discourage risk-taking

        # Special mode activation bonus (if active compared to previous)
        if game_wrapper.special_mode_active and not prev_state.get('prev_special_mode_active', False):
            special_mode_reward = 2.0
            additional_reward += special_mode_reward
            reward_components["special_mode_active"] = special_mode_reward
        
        # Combine all rewards with appropriate balancing
        total_reward = score_reward + additional_reward + saver_penalty

        return total_reward


    @staticmethod
    def progressive(current_fitness, previous_fitness, game_wrapper, frames_played=0, prev_state=None):
        """
        Progressive reward shaping that adapts based on agent skill level.
        Rewards shift from survival to basic scoring to advanced mechanics as agent improves.
        """
        # Initialize tracking state if needed
        if prev_state is None:
            prev_state = {
                'skill_phase': 0,  # 0=survival, 1=scoring, 2=mechanics, 3=mastery
                'best_score': 0,
                'games_played': 0,
                'prev_caught': 0,
                'phase_threshold': [5000, 50000, 200000],  # Score thresholds for phase transitions
                'consecutive_survivals': 0
            }
        
        # Phase transition logic
        if current_fitness > prev_state.get('best_score', 0):
            prev_state['best_score'] = current_fitness
            
        # Determine current skill phase based on best score and survival rate
        skill_phase = prev_state.get('skill_phase', 0)
        
        # Get phase_threshold with a default value if not found
        phase_threshold = prev_state.get('phase_threshold', [5000, 50000, 200000])
        
        # Ensure we have a valid index and threshold
        threshold_index = min(2, skill_phase)
        if current_fitness > phase_threshold[threshold_index]:
            skill_phase = min(3, skill_phase + 1)
            prev_state['skill_phase'] = skill_phase
            
        # Basic score difference, logarithmically scaled
        score_diff = current_fitness - previous_fitness
        if score_diff > 0:
            import numpy as np
            # Adaptive scaling based on skill phase
            score_scale = [0.001, 0.005, 0.01, 0.02][skill_phase]
            score_reward = score_scale * np.log(1 + score_diff)
        else:
            score_reward = 0
            
        # Ball survival rewards - emphasized in early phases, reduced in later phases
        survival_importance = [1.0, 0.6, 0.3, 0.1][skill_phase]
        ball_alive_reward = 0.1 * survival_importance
        
        # Game mechanics rewards - minimal in early phases, emphasized in later phases
        mechanics_importance = [0.2, 0.6, 1.0, 1.2][skill_phase]
        
        # Calculate mechanics rewards with progressive scaling
        mechanics_reward = 0
        
        # Pokemon catch reward (with progressive scaling)
        if game_wrapper.pokemon_caught_in_session > prev_state.get('prev_caught', 0):
            catch_base_reward = 2.0
            mechanics_reward += catch_base_reward * mechanics_importance
        
        # Add other mechanics with progressive scaling...
        # (Evolution, stage completion, ball upgrades, etc.)
        
        # Final reward is phase-appropriate weighted combination
        total_reward = score_reward + ball_alive_reward + (mechanics_reward * mechanics_importance)
        
        # Add metadata to track phases in the logs
        reward_metadata = {
            'skill_phase': skill_phase,
            'score_component': score_reward,
            'survival_component': ball_alive_reward,
            'mechanics_component': mechanics_reward
        }
        
        return total_reward
"""Info building utilities."""

from typing import Dict, Any

class InfoBuilder:
    """Builds a minimal info dict for the Pokemon Pinball env."""

    @staticmethod
    def build_info(game_wrapper, fitness: float, frames_played: int,
                   episodes_completed: int, episode_complete: bool = False
                  ) -> Dict[str, Any]:
        """
        Args:
            game_wrapper: PyBoy wrapper
            fitness: agent’s current fitness/score
            frames_played: frames so far in the episode
            episodes_completed: total episodes finished
            episode_complete: did we just end an episode?
        """
        # core metrics
        info = {
            'score': game_wrapper.score,
            'fitness': fitness,
            'frames_played': frames_played,
            'episodes_completed': episodes_completed,
            'episode_complete': episode_complete,
        }

        # aggregate progress
        total_stages = sum([
            game_wrapper.diglett_stages_completed,
            game_wrapper.gengar_stages_completed,
            game_wrapper.meowth_stages_completed,
            game_wrapper.seel_stages_completed,
            game_wrapper.mewtwo_stages_completed,
        ])
        total_upgrades = sum([
            game_wrapper.great_ball_upgrades,
            game_wrapper.ultra_ball_upgrades,
            game_wrapper.master_ball_upgrades,
        ])

        info['total_stages_completed'] = total_stages
        info['total_ball_upgrades'] = total_upgrades

        return info
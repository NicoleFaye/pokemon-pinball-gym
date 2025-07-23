"""Info building utilities."""

from typing import Dict, Any

class InfoBuilder:
    """Builds a minimal info dict for the Pokemon Pinball env."""

    @staticmethod
    def build_info(game_wrapper ) -> Dict[str, Any]:
        """
        Args:
            game_wrapper: PyBoy wrapper
            fitness: agent’s current fitness/score
        """
        # core metrics
        info = {
            'score': game_wrapper.score,
            'pokemon_seen': game_wrapper.pokemon_seen_in_session,
            'pokemon_caught': game_wrapper.pokemon_caught_in_session,
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
        info['lost_ball_during_saver']= game_wrapper.lost_ball_during_saver

        return info
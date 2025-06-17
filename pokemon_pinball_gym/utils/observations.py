"""Observation building utilities."""

from typing import Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy.plugins.game_wrapper_pokemon_pinball import Stage, BallType, SpecialMode


# Build mappings between enums and sequential indices
STAGE_ENUMS = list(Stage)
STAGE_TO_INDEX = {stage: idx for idx, stage in enumerate(STAGE_ENUMS)}

BALL_TYPE_ENUMS = list(BallType)
BALL_TYPE_TO_INDEX = {ball_type: idx for idx, ball_type in enumerate(BALL_TYPE_ENUMS)}


class ObservationBuilder:
    """Builds observations based on configuration and game state."""
    
    def __init__(self, config):
        """
        Initialize observation builder.
        
        Args:
            config: EnvironmentConfig instance
        """
        self.config = config
        self.info_level = config.info_level
        self.visual_mode = config.visual_mode
        self.reduce_screen_resolution = config.reduce_screen_resolution
        
        # Set output shape based on visual mode
        if self.visual_mode == "game_area":
            self.output_shape = (16, 20)
        else:  # screen mode
            if self.reduce_screen_resolution:
                self.output_shape = (72, 80)
            else:
                self.output_shape = (144, 160)
        
        # Add missing 3rd dimension
        if config.grayscale:
            self.output_shape += (1,) 
        else: 
            self.output_shape += (3,)
    
    def create_observation_space(self) -> gym.spaces.Space:
        """Create observation space based on info level."""
        observations_dict = {}
        
        # Base visual observation
        observations_dict['visual_representation'] = spaces.Box(
            low=0, high=255, shape=self.output_shape, dtype=np.uint8
        )
        
        if self.info_level == 0:
            return observations_dict['visual_representation']
        
        # Level 1: Ball position and velocity
        if self.info_level >= 1:
            obs_shape = (1,)
            observations_dict.update({
                'ball_x': spaces.Box(low=-128, high=128, shape=obs_shape, dtype=np.float32),
                'ball_y': spaces.Box(low=-128, high=128, shape=obs_shape, dtype=np.float32),
                'ball_x_velocity': spaces.Box(low=-128, high=128, shape=obs_shape, dtype=np.float32),
                'ball_y_velocity': spaces.Box(low=-128, high=128, shape=obs_shape, dtype=np.float32),
            })
        
        # Level 2: Game state
        if self.info_level >= 2:
            observations_dict.update({
                'current_stage': spaces.Discrete(len(STAGE_ENUMS)),
                'ball_type': spaces.Discrete(len(BALL_TYPE_ENUMS)),
                'special_mode': spaces.Discrete(len(SpecialMode)),
                'special_mode_active': spaces.Discrete(2),
                'saver_active': spaces.Discrete(2),
            })
        
        # Level 3: Detailed information
        if self.info_level >= 3:
            observations_dict.update({
                'pikachu_saver_charge': spaces.Discrete(16),
            })
        
        return spaces.Dict(observations_dict)

    def get_visual_observation(self, pyboy, game_wrapper):
        """Get visual observation based on visual mode."""
        if self.visual_mode == "game_area":
            game_area = game_wrapper.game_area()
            return game_area + (1,)
            
        else:
            # Get screen RGB
            screen_img = np.array(pyboy.screen.ndarray[:, :, :3], copy=True)
            
            # Reduce resolution if needed (do this on RGB first)
            if self.reduce_screen_resolution:
                screen_img = screen_img[::2, ::2]
            
            if self.config.grayscale:
                # Convert to grayscale and add channel dimension
                screen_img = np.mean(screen_img, axis=2, keepdims=True).astype(np.uint8)
                # Result: (72, 80, 1) or (144, 160, 1)
            else:
                # Keep as RGB
                screen_img = screen_img.astype(np.uint8)
                # Result: (72, 80, 3) or (144, 160, 3)

            return screen_img

    def build_observation(self, pyboy, game_wrapper) -> Dict[str, np.ndarray]:
        """Build complete observation dictionary."""
        visual_obs = self.get_visual_observation(pyboy, game_wrapper)
        observation = {
            "visual_representation": np.asarray(visual_obs, dtype=np.uint8),
        }
        
        if self.info_level == 0:
            return observation.get('visual_representation')
        
        # Add ball information
        observation.update({
            "ball_x": np.array([float(game_wrapper.ball_x)], dtype=np.float32),
            "ball_y": np.array([float(game_wrapper.ball_y)], dtype=np.float32),
            "ball_x_velocity": np.array([float(game_wrapper.ball_x_velocity)], dtype=np.float32),
            "ball_y_velocity": np.array([float(game_wrapper.ball_y_velocity)], dtype=np.float32),
        })
        
        if self.info_level == 1:
            return observation
        
        # Add game state information
        if self.info_level >= 2:
            current_stage_idx = STAGE_TO_INDEX.get(game_wrapper.current_stage, 0)
            ball_type_idx = BALL_TYPE_TO_INDEX.get(game_wrapper.ball_type, 0)
            
            observation.update({
                "current_stage": np.array([current_stage_idx], dtype=np.int32),
                "ball_type": np.array([ball_type_idx], dtype=np.int32),
                "special_mode": np.array([int(game_wrapper.special_mode)], dtype=np.int32),
                "special_mode_active": np.array([int(game_wrapper.special_mode_active)], dtype=np.int32),
                "saver_active": np.array([int(game_wrapper.ball_saver_seconds_left > 0)], dtype=np.int32),
            })
        
        # Add detailed information
        if self.info_level >= 3:
            observation["pikachu_saver_charge"] = np.array([int(game_wrapper.pikachu_saver_charge)], dtype=np.int32)
        
        return observation
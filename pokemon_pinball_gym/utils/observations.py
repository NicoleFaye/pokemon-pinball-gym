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

PYBOY_OUTPUT_HEIGHT = 144
PYBOY_OUTPUT_WIDTH = 160

PYBOY_GAME_AREA_HEIGHT = 16
PYBOY_GAME_AREA_WIDTH = 20


class ObservationBuilder:
    """Builds observations based on configuration and game state."""
    
    def __init__(self, config, pyboy):
        """
        Initialize observation builder.
        
        Args:
            config: EnvironmentConfig instance
        """
        self.pyboy = pyboy
        self.config = config
        self.observation_info_level = config.observation_info_level
        self.visual_mode = config.visual_mode
        self.n_frame_stack = config.frame_stack
        self.reduce_screen_resolution = config.reduce_screen_resolution
        #initialize empty array for frame stacking using np of size n_frame_stack
        height = PYBOY_OUTPUT_HEIGHT // 2 if self.reduce_screen_resolution else PYBOY_OUTPUT_HEIGHT
        width = PYBOY_OUTPUT_WIDTH // 2 if self.reduce_screen_resolution else PYBOY_OUTPUT_WIDTH
        self.render_frame_stack = np.zeros((height, width, self.n_frame_stack), dtype=np.uint8)
        # Set output shape based on visual mode
        if self.visual_mode == "game_area":
            self.output_shape = (PYBOY_GAME_AREA_HEIGHT, PYBOY_GAME_AREA_WIDTH, self.n_frame_stack)  
        else:  
            self.output_shape = (height, width, self.n_frame_stack)

        self.coord_frame_stack = np.zeros((2,self.n_frame_stack), dtype=np.float32)
        self.vel_frame_stack = np.zeros((2,self.n_frame_stack), dtype=np.float32)  
            
    def create_observation_space(self) -> gym.spaces.Space:
        """Create observation space based on info level."""
        observations_dict = {}
        
        # Base visual observation
        observations_dict['visual_representation'] = spaces.Box(
            low=0, high=255, shape=self.output_shape, dtype=np.uint8
        )
        
        if self.observation_info_level == 0:
            return observations_dict['visual_representation']
        
        # ignore below this point for now, as we are not using it

        obs_shape = (1,self.n_frame_stack)
        # Level 1: Ball position
        if self.observation_info_level >= 1:
            observations_dict.update({
                'coords': spaces.Box(low=-128, high=128, shape=(2, self.n_frame_stack), dtype=np.float32),
            })
        # Level 2: Ball velocity
        if self.observation_info_level >= 2:
            observations_dict.update({
                'velocity': spaces.Box(low=-128, high=128, shape=(2, self.n_frame_stack), dtype=np.float32),
            })
        
        # Level 3: Game state information
        if self.observation_info_level >= 3:
            observations_dict.update({
                'current_stage': spaces.Discrete(len(STAGE_ENUMS)),
                'ball_type': spaces.Discrete(len(BALL_TYPE_ENUMS)),
                'special_mode': spaces.Discrete(len(SpecialMode)),
                'special_mode_active': spaces.Discrete(2),
                'saver_active': spaces.Discrete(2),
            })
        
        # Level 3: Detailed information
        if self.observation_info_level >= 4:
            observations_dict.update({
                'pikachu_saver_charge': spaces.Discrete(16),
            })
        
        return spaces.Dict(observations_dict)

    def render(self):
        screen = np.expand_dims(self.pyboy.screen.ndarray[:, :, 1], axis=-1)
        screen = screen[::2, ::2]
        return screen

    def build_observation(self, pyboy, game_wrapper) :#-> Dict[str, np.ndarray]:
        """Build complete observation dictionary."""
        
        #roll observation onto frame stack and add new frame using render
        self.render_frame_stack = np.roll(self.render_frame_stack, shift=-1, axis=-1)
        self.render_frame_stack[:,:,-1:] = self.render()
        if self.observation_info_level == 0:
            return self.render_frame_stack

        observation = {
            "visual_representation": self.render_frame_stack
        }         

        self.coord_frame_stack = np.roll(self.coord_frame_stack, shift=-1, axis=-1)
        self.coord_frame_stack[0, -1] = float(game_wrapper.ball_x)
        self.coord_frame_stack[1, -1] = float(game_wrapper.ball_y)

        # Add ball information
        observation.update({
            "coords": self.coord_frame_stack
        })
        
        
        if self.observation_info_level >= 2:
            self.vel_frame_stack = np.roll(self.vel_frame_stack, shift=-1, axis=-1)
            self.vel_frame_stack[0, -1] = float(game_wrapper.ball_x_velocity)
            self.vel_frame_stack[1, -1] = float(game_wrapper.ball_y_velocity)
            observation.update({
                "velocity": self.vel_frame_stack
            })

        # Add game state information
        if self.observation_info_level >= 3:
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
        if self.observation_info_level >= 4:
            observation["pikachu_saver_charge"] = np.array([int(game_wrapper.pikachu_saver_charge)], dtype=np.int32)
        
        return observation
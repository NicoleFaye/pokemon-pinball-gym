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

COLOR_CHANNEL_INDEX = 1


class ObservationBuilder:
    """Builds observations based on configuration and game state."""

    def __init__(self, config, pyboy):
        self.pyboy = pyboy
        self.config = config
        self.observation_info_level = config.observation_info_level
        self.visual_mode = config.visual_mode
        self.n_frame_stack = config.frame_stack
        self.reduce_screen_resolution = config.reduce_screen_resolution

        height = PYBOY_OUTPUT_HEIGHT // 2 if self.reduce_screen_resolution else PYBOY_OUTPUT_HEIGHT
        width = PYBOY_OUTPUT_WIDTH // 2 if self.reduce_screen_resolution else PYBOY_OUTPUT_WIDTH
        visual_shape = (height, width, self.n_frame_stack)

        self.output_shape = (PYBOY_GAME_AREA_HEIGHT, PYBOY_GAME_AREA_WIDTH, self.n_frame_stack) \
            if self.visual_mode == "game_area" else visual_shape

        self.frame_stacks: Dict[str, np.ndarray] = {
            "visual_representation": np.zeros(self.output_shape, dtype=np.uint8)
        }

        if self.observation_info_level >= 2:
            self.frame_stacks["coords"] = np.zeros((2, self.n_frame_stack), dtype=np.float32)

        if self.observation_info_level >= 3:
            self.frame_stacks["velocity"] = np.zeros((2, self.n_frame_stack), dtype=np.float32)

    def create_observation_space(self) -> gym.spaces.Space:
        """Create observation space based on info level."""
        observations_dict = {}

        observations_dict['visual_representation'] = spaces.Box(
            low=0, high=255, shape=self.output_shape, dtype=np.uint8
        )

        if self.observation_info_level == 0:
            return observations_dict['visual_representation']

        if self.observation_info_level >= 1:
            observations_dict.update({
                'current_stage': spaces.Discrete(9),
            })

        if self.observation_info_level >= 2:
            observations_dict['coords'] = spaces.Box(low=-128, high=128, shape=(2, self.n_frame_stack), dtype=np.float32)

        if self.observation_info_level >= 3:
            observations_dict['velocity'] = spaces.Box(low=-128, high=128, shape=(2, self.n_frame_stack), dtype=np.float32)

        if self.observation_info_level >= 4:
            observations_dict.update({
                'current_stage': spaces.Discrete(len(STAGE_ENUMS)),
                'ball_type': spaces.Discrete(len(BALL_TYPE_ENUMS)),
                'special_mode': spaces.Discrete(len(SpecialMode)),
                'special_mode_active': spaces.Discrete(2),
                'saver_active': spaces.Discrete(2),
            })

        if self.observation_info_level >= 5:
            observations_dict['pikachu_saver_charge'] = spaces.Discrete(16)

        return spaces.Dict(observations_dict)

    def render(self):
        screen = np.expand_dims(self.pyboy.screen.ndarray[:, :, COLOR_CHANNEL_INDEX], axis=-1)
        screen = screen[::2, ::2]
        return screen

    def _update_stack(self, key: str, value: np.ndarray):
        stack = self.frame_stacks[key]
        self.frame_stacks[key] = np.roll(stack, shift=-1, axis=-1)
        self.frame_stacks[key][..., -1] = value

    def build_observation(self, pyboy, game_wrapper):
        """Build complete observation dictionary."""

        rendered_frame = self.render()
        self._update_stack("visual_representation", rendered_frame.squeeze())

        if self.observation_info_level == 0:
            return self.frame_stacks["visual_representation"]

        observation = {"visual_representation": self.frame_stacks["visual_representation"]}

        if self.observation_info_level >= 1:
            stage_idx = STAGE_TO_INDEX.get(game_wrapper.current_stage, 0)
            observation["current_stage"] = np.array([stage_idx], dtype=np.int32)

        if self.observation_info_level >= 2:
            coords = np.array([game_wrapper.ball_x, game_wrapper.ball_y], dtype=np.float32)
            self._update_stack("coords", coords)
            observation["coords"] = self.frame_stacks["coords"]

        if self.observation_info_level >= 3:
            velocity = np.array([game_wrapper.ball_x_velocity, game_wrapper.ball_y_velocity], dtype=np.float32)
            self._update_stack("velocity", velocity)
            observation["velocity"] = self.frame_stacks["velocity"]

        if self.observation_info_level >= 4:
            observation.update({
                "current_stage": np.array([stage_idx], dtype=np.int32),
                "ball_type": np.array([BALL_TYPE_TO_INDEX.get(game_wrapper.ball_type, 0)], dtype=np.int32),
                "special_mode": np.array([int(game_wrapper.special_mode)], dtype=np.int32),
                "special_mode_active": np.array([int(game_wrapper.special_mode_active)], dtype=np.int32),
                "saver_active": np.array([int(game_wrapper.ball_saver_seconds_left > 0)], dtype=np.int32),
            })

        if self.observation_info_level >= 5:
            observation["pikachu_saver_charge"] = np.array([int(game_wrapper.pikachu_saver_charge)], dtype=np.int32)

        return observation

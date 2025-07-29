from typing import Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy.plugins.game_wrapper_pokemon_pinball import Stage, BallType, SpecialMode

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
    def __init__(self, config, pyboy):
        self.pyboy = pyboy
        self.config = config
        self.observation_info_level = config.observation_info_level
        self.visual_mode = config.visual_mode
        self.n_frame_stack = config.frame_stack
        self.reduce_screen_resolution = config.reduce_screen_resolution

        h = PYBOY_OUTPUT_HEIGHT // 2 if self.reduce_screen_resolution else PYBOY_OUTPUT_HEIGHT
        w = PYBOY_OUTPUT_WIDTH // 2 if self.reduce_screen_resolution else PYBOY_OUTPUT_WIDTH
        self.visual_shape = (h, w, self.n_frame_stack)

        self.frame_stacks: Dict[str, np.ndarray] = {
            "visual_representation": np.zeros(self.visual_shape, dtype=np.uint8)
        }

        if self.observation_info_level >= 1:
            self.frame_stacks["current_stage"] = np.zeros((self.n_frame_stack,), dtype=np.int32)

        if self.observation_info_level >= 2:
            self.frame_stacks["coords"] = np.zeros((2, self.n_frame_stack), dtype=np.float32)

        if self.observation_info_level >= 3:
            self.frame_stacks["velocity"] = np.zeros((2, self.n_frame_stack), dtype=np.float32)

        self.extra_features = []

        if self.observation_info_level >= 4:
            self.extra_features += [
                ('ball_type', len(BALL_TYPE_ENUMS)),
                ('special_mode', len(SpecialMode)),
                ('special_mode_active', 2),
                ('saver_active', 2)
            ]

        if self.observation_info_level >= 5:
            self.extra_features += [('pikachu_saver_charge', 16)]

    def create_observation_space(self) -> gym.spaces.Space:
        vis_dim = np.prod(self.visual_shape)
        stacked_dim = 0

        if self.observation_info_level >= 1:
            stacked_dim += self.n_frame_stack
        if self.observation_info_level >= 2:
            stacked_dim += 2 * self.n_frame_stack
        if self.observation_info_level >= 3:
            stacked_dim += 2 * self.n_frame_stack
        if self.observation_info_level >= 4:
            stacked_dim += 4
        if self.observation_info_level >= 5:
            stacked_dim += 1

        total_dim = vis_dim + stacked_dim
        return spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

    def render(self):
        screen = np.expand_dims(self.pyboy.screen.ndarray[:, :, COLOR_CHANNEL_INDEX], axis=-1)
        screen = screen[::2, ::2]
        return screen

    def _update_stack(self, key: str, value: np.ndarray):
        stack = self.frame_stacks[key]
        self.frame_stacks[key] = np.roll(stack, shift=-1, axis=-1 if stack.ndim > 1 else 0)
        if stack.ndim > 1:
            self.frame_stacks[key][..., -1] = value
        else:
            self.frame_stacks[key][-1] = value

    def build_observation(self, game_wrapper):
        self._update_stack("visual_representation", self.render().squeeze())

        features = [self.frame_stacks["visual_representation"].flatten().astype(np.float32)]

        if self.observation_info_level >= 1:
            stage_idx = STAGE_TO_INDEX.get(game_wrapper.current_stage, 0)
            self._update_stack("current_stage", np.array(stage_idx, dtype=np.int32))
            features.append(self.frame_stacks["current_stage"].astype(np.float32))

        if self.observation_info_level >= 2:
            coords = np.array([game_wrapper.ball_x, game_wrapper.ball_y], dtype=np.float32)
            self._update_stack("coords", coords)
            features.append(self.frame_stacks["coords"].astype(np.float32))

        if self.observation_info_level >= 3:
            velocity = np.array([game_wrapper.ball_x_velocity, game_wrapper.ball_y_velocity], dtype=np.float32)
            self._update_stack("velocity", velocity)
            features.append(self.frame_stacks["velocity"].astype(np.float32))

        if self.observation_info_level >= 4:
            features.extend([
                np.array([BALL_TYPE_TO_INDEX.get(game_wrapper.ball_type, 0)], dtype=np.float32),
                np.array([int(game_wrapper.special_mode)], dtype=np.float32),
                np.array([int(game_wrapper.special_mode_active)], dtype=np.float32),
                np.array([int(game_wrapper.ball_saver_seconds_left > 0)], dtype=np.float32)
            ])

        if self.observation_info_level >= 5:
            features.append(np.array([int(game_wrapper.pikachu_saver_charge)], dtype=np.float32))

        return np.concatenate(features, dtype=np.float32)

"""Pokemon Pinball Gymnasium environment."""
import enum
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import Stage, BallType, SpecialMode, Maps, Pokemon

from ..rewards import RewardFunction, BasicReward, CatchFocusedReward, ComprehensiveReward
from ..utils import GameStateTracker, ObservationBuilder, InfoBuilder


# Build mappings between enums and sequential indices
STAGE_ENUMS = list(Stage)
STAGE_TO_INDEX = {stage: idx for idx, stage in enumerate(STAGE_ENUMS)}
INDEX_TO_STAGE = {idx: stage for idx, stage in enumerate(STAGE_ENUMS)}

BALL_TYPE_ENUMS = list(BallType)
BALL_TYPE_TO_INDEX = {ball_type: idx for idx, ball_type in enumerate(BALL_TYPE_ENUMS)}
INDEX_TO_BALL_TYPE = {idx: ball_type for idx, ball_type in enumerate(BALL_TYPE_ENUMS)}


class Actions(enum.Enum):
    """Available actions in the Pokemon Pinball environment."""
    IDLE = 0
    LEFT_FLIPPER_PRESS = 1
    RIGHT_FLIPPER_PRESS = 2
    LEFT_FLIPPER_RELEASE = 3
    RIGHT_FLIPPER_RELEASE = 4
    #LEFT_TILT = 5
    #RIGHT_TILT = 6
    #UP_TILT = 7
    #LEFT_UP_TILT = 8
    #RIGHT_UP_TILT = 9


@dataclass
class EnvironmentConfig:
    """Configuration class for Pokemon Pinball environment."""
    debug: bool = False
    headless: bool = True
    info_level: int = 0
    frame_skip: int = 4
    visual_mode: str = "screen"  # "screen" or "game_area"
    reduce_screen_resolution: bool = True
    episode_mode: str = "ball"  # "life", "ball", or "game"
    reset_condition: str = "game"  # "life", "ball", or "game"
    grayscale: bool = False

    @classmethod
    def from_dict(cls, config_dict):
        """Create EnvironmentConfig from dictionary, ignoring extra keys."""
        from dataclasses import fields
        valid_fields = {field.name for field in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_config)


class RenderWrapper(gym.Wrapper):
    """Wrapper to add rendering capability."""
    
    def __init__(self, env):
        super().__init__(env)

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self):
        return self.env.screen.screen_ndarray()

class PokemonPinballEnv(gym.Env):
    """Pokemon Pinball environment for reinforcement learning."""
    
    metadata = {"render_modes": ["human"]}
    instance_count = 0
    
    def __init__(self, rom_path="pokemon_pinball.gbc", config=None, reward_function=None):
        """
        Initialize the Pokemon Pinball environment.
        
        Args:
            rom_path: Path to the Pokemon Pinball ROM
            config: EnvironmentConfig instance or dict
            reward_function: RewardFunction instance for calculating rewards
        """
        super().__init__()
        
        # Handle config
        if config is None:
            config = {}
        if isinstance(config, dict):
            config = EnvironmentConfig.from_dict(config)
        self.config = config

        # Handle reward function
        if reward_function is not None:
            self.reward_function = reward_function
        else:
            # Default to basic reward
            self.reward_function = ComprehensiveReward()
        
        # Validate reward function
        if not isinstance(self.reward_function, RewardFunction):
            raise TypeError("reward_function must be an instance of RewardFunction")
        
        # Instance tracking
        PokemonPinballEnv.instance_count += 1
        self.instance_id = PokemonPinballEnv.instance_count
        
        # Initialize PyBoy
        self._init_pyboy(rom_path)
        
        # Initialize components
        self.state_tracker = GameStateTracker()
        self.obs_builder = ObservationBuilder(self.config)
        
        # Initialize tracking variables
        self._init_tracking_variables()
        
        # Set up gym spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = self.obs_builder.create_observation_space()
        self.single_action_space = spaces.Discrete(len(Actions))
        self.single_observation_space = self.obs_builder.create_observation_space()
        
        # Start the game
        self._game_wrapper.start_game()
        
        # Set emulation speed
        speed = 1.0 if self.config.debug else 0
        self.pyboy.set_emulation_speed(speed)
    
    def _init_pyboy(self, rom_path: str):
        """Initialize PyBoy instance."""
        import os
        pid = os.getpid()
        #print(f"Creating PokemonPinballEnv instance {self.instance_id} in process {pid}")
        
        window_type = "null" if self.config.headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window_type, sound_emulated=False)
        
        if self.pyboy is None:
            raise ValueError("PyBoy instance is required")
        
        if self.pyboy.cartridge_title != "POKEPINBALLVPH":
            raise ValueError("Invalid ROM: Pokémon Pinball required")
        
        self._game_wrapper = self.pyboy.game_wrapper
        self.pyboy.set_emulation_speed(0)
    
    def _init_tracking_variables(self):
        """Initialize tracking variables."""
        self._fitness = 0
        self._previous_fitness = 0
        self._frames_played = 0
        self._high_score = 0
        self._episode_count = 0
        self.episodes_completed = 0
        self._initialized = False
        
        # Stuck detection (if needed)
        self.stuck_detection_window = 100
        self.stuck_detection_threshold = 5.0
        self.stuck_detection_reward_threshold = 100
        self.ball_position_history = []
        self.last_score = 0
    
    @property
    def ball_lost(self) -> bool:
        """Check if a ball was just lost."""
        lives_decreased = self._game_wrapper.balls_left < self.state_tracker.prev_balls_left
        balls_lost_during_saver_increased = (
            self._game_wrapper.lost_ball_during_saver > self.state_tracker.prev_balls_lost_during_saver
        )
        return lives_decreased or balls_lost_during_saver_increased
    
    @property
    def life_lost(self) -> bool:
        """Check if a life was just lost."""
        return self._game_wrapper.balls_left < self.state_tracker.prev_balls_left
    
    def _calculate_fitness(self) -> float:
        """Calculate fitness/reward using the configured reward function."""
        # Update fitness tracking
        self._previous_fitness = self._fitness
        self._fitness = self._game_wrapper.score
        
        # Get game state for reward function
        game_state = self.state_tracker.to_dict()
        
        # Call the reward function
        reward = self.reward_function.calculate_reward(
            self._fitness,
            self._previous_fitness,
            self._game_wrapper,
            self._frames_played,
            game_state
        )
        
        # Update tracking state
        self.state_tracker.update(self._game_wrapper)
        
        return reward
    
    def _execute_action(self, action: int):
        """Execute the given action."""
        action_release_delay = math.ceil((1 + self.config.frame_skip) / 2)
        
        action_map = {
            Actions.LEFT_FLIPPER_PRESS.value: lambda: self.pyboy.button_press("left"),
            Actions.RIGHT_FLIPPER_PRESS.value: lambda: self.pyboy.button_press("a"),
            Actions.LEFT_FLIPPER_RELEASE.value: lambda: self.pyboy.button_release("left"),
            Actions.RIGHT_FLIPPER_RELEASE.value: lambda: self.pyboy.button_release("a"),
            #Actions.LEFT_TILT.value: lambda: self.pyboy.button("down",action_release_delay),
            #Actions.RIGHT_TILT.value: lambda: self.pyboy.button("b",action_release_delay),
            #Actions.UP_TILT.value: lambda: self.pyboy.button("select",action_release_delay),
            #Actions.LEFT_UP_TILT.value: lambda: (self.pyboy.button("select",action_release_delay), self.pyboy.button("down",action_release_delay)),
            #Actions.RIGHT_UP_TILT.value: lambda: (self.pyboy.button("select",action_release_delay), self.pyboy.button("b",action_release_delay)),
       
        }
        
        # Execute action if it's not IDLE
        if action > 0 and action < len(Actions):
            action_func = action_map.get(action)
            if action_func:
                action_func()
    
    def _is_episode_done(self) -> bool:
        """Determine if episode should end based on episode mode."""
        if self.config.episode_mode == "life":
            return self._game_wrapper.lost_ball_during_saver
        elif self.config.episode_mode == "ball":
            return self.ball_lost
        elif self.config.episode_mode == "game":
            return self._game_wrapper.game_over
        else:
            # Default behavior
            return (self._game_wrapper.lost_ball_during_saver or 
                    self._game_wrapper.game_over or 
                    self._game_wrapper.balls_left < 2)
    
    def step(self, action):
        """Take a step in the environment."""
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        
        # Save current state before taking action
        self.state_tracker.prev_balls_left = self._game_wrapper.balls_left
        
        # Execute action
        self._execute_action(action)
        
        # Advance game
        ticks = self.config.frame_skip
        if not self.config.debug:
            self.pyboy.tick(ticks, not self.config.headless, False)
        else:
            for tick in range(ticks):
                self.pyboy.tick(1, not self.config.headless, False)
        self._frames_played += ticks
        
        # Calculate reward using the configured reward function
        reward = self._calculate_fitness()
        
        # Check if episode is done
        done = self._is_episode_done()
        truncated = False
        
        # Get observation and info
        observation = self.obs_builder.build_observation(self.pyboy, self._game_wrapper)
        
        # Build info
        high_score = False
        if done and self._game_wrapper.score > self._high_score:
            self._high_score = self._game_wrapper.score
            high_score = True
        
        # Only provide info when episode ends, following Pokemon Red's pattern
        info = {}
        if done:
            info = InfoBuilder.build_info(
                self._game_wrapper, self._fitness, self._frames_played,
                self.episodes_completed, episode_complete=done
            )
        
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Handle reset condition
        self._handle_reset_condition()
        
        # Reset episode tracking
        self._fitness = 0
        self._previous_fitness = 0
        self._frames_played = 0
        
        # Reset detection
        self.ball_position_history = []
        self.last_score = 0
        
        # Update counters
        self.episodes_completed += 1
        self._episode_count += 1
        self._initialized = True
        
        # Update state tracker
        self.state_tracker.prev_balls_left = self._game_wrapper.balls_left
        self.state_tracker.prev_balls_lost_during_saver = self._game_wrapper.lost_ball_during_saver
        
        # Get initial observation and info
        observation = self.obs_builder.build_observation(self.pyboy, self._game_wrapper)
        
        return observation, {}
    
    def _handle_reset_condition(self):
        """Handle reset based on reset condition configuration."""
        game_wrapper = self._game_wrapper
        game_wrapper.reset_tracking()
        
        if self.config.reset_condition == "life":
            if game_wrapper.game_over or self.life_lost:
                game_wrapper.reset_game()
                self.state_tracker.reset()
        elif self.config.reset_condition == "ball":
            if game_wrapper.game_over or self.ball_lost:
                game_wrapper.reset_game()
                self.state_tracker.reset()
        elif self.config.reset_condition == "game":
            if game_wrapper.game_over:
                game_wrapper.reset_game()
                self.state_tracker.reset()
    
    def render(self, mode="human"):
        """Render the environment."""
        pass
    
    def close(self):
        """Close the environment."""
        self.pyboy.stop()
        PokemonPinballEnv.instance_count = max(0, PokemonPinballEnv.instance_count - 1)
    
    @classmethod
    def get_instance_count(cls):
        """Get the current number of environment instances."""
        return cls.instance_count
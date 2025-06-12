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
from rewards import RewardShaping
import functools


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
    headless: bool = False
    reward_shaping: str = "comprehensive"
    info_level: int = 0
    frame_skip: int = 4
    visual_mode: str = "screen"  # "screen" or "game_area"
    reduce_screen_resolution: bool = True
    episode_mode: str = "life"  # "life", "ball", or "game", where ball triggers even if the ball saver is active
    reset_condition: str = "game"  # "life", "ball", or "game"

    @classmethod
    def from_dict(cls, config_dict):
        """Create EnvironmentConfig from dictionary, ignoring extra keys."""
        from dataclasses import fields
        valid_fields = {field.name for field in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_config)


class GameStateTracker:
    """Tracks game state changes for reward calculation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracking variables."""
        self.prev_caught = 0
        self.prev_evolutions = 0
        self.prev_stages_completed = 0
        self.prev_ball_upgrades = 0
        self.prev_balls_left = 2
        self.prev_balls_lost_during_saver = 0
    
    def to_dict(self) -> Dict[str, int]:
        """Convert state to dictionary for reward functions."""
        return {
            'prev_caught': self.prev_caught,
            'prev_evolutions': self.prev_evolutions,
            'prev_stages_completed': self.prev_stages_completed,
            'prev_ball_upgrades': self.prev_ball_upgrades,
            'prev_ball_lost_during_saver': self.prev_balls_lost_during_saver,
        }
    
    def update(self, game_wrapper):
        """Update tracking state based on current game wrapper state."""
        self.prev_caught = game_wrapper.pokemon_caught_in_session
        self.prev_evolutions = game_wrapper.evolution_success_count
        
        self.prev_stages_completed = (
            game_wrapper.diglett_stages_completed +
            game_wrapper.gengar_stages_completed +
            game_wrapper.meowth_stages_completed +
            game_wrapper.seel_stages_completed +
            game_wrapper.mewtwo_stages_completed
        )
        
        self.prev_ball_upgrades = (
            game_wrapper.great_ball_upgrades +
            game_wrapper.ultra_ball_upgrades +
            game_wrapper.master_ball_upgrades
        )
        
        self.prev_balls_left = game_wrapper.balls_left
        self.prev_balls_lost_during_saver = game_wrapper.lost_ball_during_saver


class ObservationBuilder:
    """Builds observations based on configuration and game state."""
    
    def __init__(self, config: EnvironmentConfig):
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
            return game_wrapper.game_area()
        else:
            # Get screen and convert to grayscale
            screen_img = np.array(pyboy.screen.ndarray[:, :, :3], copy=True)
            screen_img = np.mean(screen_img, axis=2, keepdims=False).astype(np.uint8)
            
            if self.reduce_screen_resolution:
                screen_img = screen_img[::2, ::2]
            
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


class InfoBuilder:
    """Builds info dictionary for environment feedback."""
    
    @staticmethod
    def build_info(game_wrapper, fitness, frames_played, episodes_completed, episode_count, 
                   episode_mode, reset_condition, episode_complete=False, high_score=False) -> Dict[str, Any]:
        """Build comprehensive info dictionary."""
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

class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)  # This is crucial!

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self):
        return self.env.screen.screen_ndarray()

class PokemonPinballEnv(gym.Env):
    """Pokemon Pinball environment for reinforcement learning."""
    
    metadata = {"render_modes": ["human"]}
    instance_count = 0
    
    def __init__(self, rom_path="pokemon_pinball.gbc", config=None):
        """Initialize the Pokemon Pinball environment."""
        super().__init__()
        
        # Handle config
        if config is None:
            config = {}
        if isinstance(config, dict):
            config = EnvironmentConfig.from_dict(config)
        self.config = config

        #self.emulated = True
        #self.num_agents = config.num_agents
        
        # Instance tracking
        PokemonPinballEnv.instance_count += 1
        self.instance_id = PokemonPinballEnv.instance_count
        
        # Initialize PyBoy
        self._init_pyboy(rom_path)
        
        # Initialize components
        self.state_tracker = GameStateTracker()
        self.obs_builder = ObservationBuilder(self.config)
        
        # Initialize reward shaping
        self.reward_shaping = self._get_reward_function(config.reward_shaping)
        
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
        print(f"Creating PokemonPinballEnv instance {self.instance_id} in process {pid}")
        
        window_type = "null" if self.config.headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window_type, sound_emulated=False)
        
        if self.pyboy is None:
            raise ValueError("PyBoy instance is required")
        
        if self.pyboy.cartridge_title != "POKEPINBALLVPH":
            raise ValueError("Invalid ROM: Pokémon Pinball required")
        
        self._game_wrapper = self.pyboy.game_wrapper
        self.pyboy.set_emulation_speed(0)
    
    def _get_reward_function(self, reward_name: str):
        """Get reward shaping function by name."""
        reward_functions = {
            'basic': RewardShaping.basic,
            'catch_focused': RewardShaping.catch_focused,
            'comprehensive': RewardShaping.comprehensive,
            'progressive': RewardShaping.progressive,
        }
        return reward_functions.get(reward_name)
    
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
    
    def _calculate_fitness(self, reward_shaping_func=None) -> float:
        """Calculate fitness/reward with optional reward shaping."""
        # Update fitness tracking
        self._previous_fitness = self._fitness
        self._fitness = self._game_wrapper.score
        
        # If no reward shaping, return simple score difference
        if reward_shaping_func is None:
            return self._fitness - self._previous_fitness
        
        # Apply reward shaping with game state tracking
        game_state = self.state_tracker.to_dict()
        
        reward = reward_shaping_func(
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
        
        # Calculate reward
        reward = self._calculate_fitness(self.reward_shaping)
        
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
        
        info = InfoBuilder.build_info(
            self._game_wrapper, self._fitness, self._frames_played,
            self.episodes_completed, self._episode_count,
            self.config.episode_mode, self.config.reset_condition,
            episode_complete=done, high_score=high_score
        )
        
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset tracking
        self.state_tracker.reset()
        
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
        info = InfoBuilder.build_info(
            self._game_wrapper, self._fitness, self._frames_played,
            self.episodes_completed, self._episode_count,
            self.config.episode_mode, self.config.reset_condition
        )
        
        return observation, info
    
    def _handle_reset_condition(self):
        """Handle reset based on reset condition configuration."""
        game_wrapper = self._game_wrapper
        game_wrapper.reset_tracking()
        
        if self.config.reset_condition == "life":
            if game_wrapper.game_over or self.life_lost:
                game_wrapper.reset_game()
        elif self.config.reset_condition == "ball":
            if game_wrapper.game_over or self.ball_lost:
                game_wrapper.reset_game()
        elif self.config.reset_condition == "game":
            if game_wrapper.game_over:
                game_wrapper.reset_game()
    
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
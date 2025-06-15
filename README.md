# Pokemon Pinball Gym

> **⚠️ Work in Progress**: This project is currently under active development. Features and APIs may change without notice. Use at your own discretion for research and experimentation.

A Gymnasium environment for Pokemon Pinball (Game Boy Color) designed for reinforcement learning research. This environment provides a flexible interface for training RL agents to play Pokemon Pinball with configurable reward functions, observation spaces, and episode termination conditions.

## Features

- **Multiple Reward Functions**: Built-in reward functions focusing on different objectives (scoring, Pokemon catching, comprehensive gameplay)
- **Configurable Observations**: Support for visual-only or multi-modal observations with varying information levels
- **Flexible Episode Management**: Customizable episode termination and reset conditions
- **Game State Tracking**: Comprehensive tracking of game progress, Pokemon catches, evolutions, and stage completions
- **Performance Optimized**: Frame skipping and resolution reduction options for faster training

## Installation

### Prerequisites

- Python 3.8+
- Pokemon Pinball ROM file (`.gbc` format)

### Install from Source

```bash
git clone https://github.com/NicoleFaye/pokemon-pinball-gym.git
cd pokemon-pinball-gym
pip install -e .
```

### Dependencies

The environment requires:
- `gymnasium>=0.29.1`
- `pyboy>=1.4.4`

## Quick Start

```python
import pokemon_pinball_gym
from pokemon_pinball_gym import PokemonPinballEnv, EnvironmentConfig, ComprehensiveReward

# Basic usage
env = PokemonPinballEnv(rom_path="path/to/pokemon_pinball.gbc")

# With custom configuration
config = EnvironmentConfig(
    headless=True,
    frame_skip=4,
    info_level=2,
    episode_mode="ball"
)

reward_fn = ComprehensiveReward(score_scale=0.3, catch_bonus=5.0)
env = PokemonPinballEnv(
    rom_path="path/to/pokemon_pinball.gbc",
    config=config,
    reward_function=reward_fn
)

# Standard gym interface
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, info = env.reset()

env.close()
```

## Configuration Options

### EnvironmentConfig

Configure environment behavior through the `EnvironmentConfig` class:

```python
config = EnvironmentConfig(
    debug=False,                    # Enable debug mode (slower, visual)
    headless=True,                  # Run without display
    info_level=2,                   # Observation complexity (0-3)
    frame_skip=4,                   # Frames to skip per action
    visual_mode="screen",           # "screen" or "game_area"
    reduce_screen_resolution=True,  # Half screen resolution
    episode_mode="ball",            # "life", "ball", or "game"
    reset_condition="game"          # "life", "ball", or "game"
)
```

**Info Levels:**
- `0`: Visual observation only
- `1`: Visual + ball position and velocity
- `2`: Level 1 + game state (stage, ball type, special modes)
- `3`: Level 2 + detailed information (Pikachu saver charge)

**Episode Modes:**
- `"life"`: Episode ends when ball is lost during saver
- `"ball"`: Episode ends when any ball is lost
- `"game"`: Episode ends when game over

## Actions

The environment supports the following actions:

```python
from pokemon_pinball_gym import Actions

# Available actions:
Actions.IDLE                  # Do nothing
Actions.LEFT_FLIPPER_PRESS    # Press left flipper
Actions.RIGHT_FLIPPER_PRESS   # Press right flipper
Actions.LEFT_FLIPPER_RELEASE  # Release left flipper
Actions.RIGHT_FLIPPER_RELEASE # Release right flipper
```

## Reward Functions

### Built-in Reward Functions

#### BasicReward
Simple score-based reward with scaling:

```python
from pokemon_pinball_gym import BasicReward

reward_fn = BasicReward(scale_factor=0.01)
```

#### CatchFocusedReward
Emphasizes Pokemon catching over pure scoring:

```python
from pokemon_pinball_gym import CatchFocusedReward

reward_fn = CatchFocusedReward(
    score_scale=0.005,
    catch_bonus=3.0
)
```

#### ComprehensiveReward (Recommended)
Multi-objective reward balancing various game aspects:

```python
from pokemon_pinball_gym import ComprehensiveReward

reward_fn = ComprehensiveReward(
    score_scale=0.3,        # Base score scaling
    catch_bonus=5.0,        # Pokemon catch bonus
    evolution_bonus=10.0,   # Evolution bonus
    stage_bonus=8.0,        # Stage completion bonus
    ball_upgrade_bonus=3.0, # Ball upgrade bonus
    saver_penalty=-3.0      # Ball loss penalty
)
```

### Custom Reward Functions

Create custom reward functions by inheriting from `RewardFunction`:

```python
from pokemon_pinball_gym.rewards import RewardFunction

class MyCustomReward(RewardFunction):
    def calculate_reward(self, current_fitness, previous_fitness, 
                        game_wrapper, frames_played, prev_state=None):
        # Your custom reward logic here
        return reward_value
    
    def get_info(self):
        return {
            'reward_function': 'MyCustomReward',
            'description': 'My custom reward implementation'
        }
```

#### Available Game Wrapper Properties

The `game_wrapper` parameter provides access to numerous Pokemon Pinball game properties for creating custom reward functions. For a complete list of available properties and methods, reference the [PyBoy Pokemon Pinball wrapper source code](https://github.com/Baekalfen/PyBoy/blob/master/pyboy/plugins/game_wrapper_pokemon_pinball.py).

## Observations

Observation structure depends on the `info_level` setting:

### Level 0 (Visual Only)
```python
# Returns numpy array of shape (72, 80) or (144, 160)
observation = env.step(action)[0]  # Direct numpy array
```

### Level 1+ (Dictionary)
```python
observation = {
    'visual_representation': np.array(...),  # Game screen
    'ball_x': np.array([x]),                 # Ball X position
    'ball_y': np.array([y]),                 # Ball Y position
    'ball_x_velocity': np.array([vx]),       # Ball X velocity
    'ball_y_velocity': np.array([vy]),       # Ball Y velocity
    # Level 2+ adds:
    'current_stage': np.array([stage_idx]),   # Current stage index
    'ball_type': np.array([ball_idx]),        # Ball type index
    'special_mode': np.array([mode]),         # Special mode
    'special_mode_active': np.array([bool]),  # Special mode active
    'saver_active': np.array([bool]),         # Ball saver active
    # Level 3 adds:
    'pikachu_saver_charge': np.array([charge]) # Pikachu saver charge
}
```

## Info Dictionary

The environment provides comprehensive game information:

```python
info = {
    'score': [current_score],
    'pokemon_caught': [pokemon_count],
    'evolutions': [evolution_count],
    'total_stages_completed': [stage_count],
    'ball_type': ['POKEBALL' | 'GREATBALL' | 'ULTRABALL' | 'MASTERBALL'],
    'current_stage': [stage_name],
    'ball_x': [x_position],
    'ball_y': [y_position],
    'balls_left': [remaining_balls],
    # ... and many more fields
}
```



## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [PyBoy](https://github.com/Baekalfen/PyBoy) for Game Boy emulation
- Uses [Gymnasium](https://gymnasium.farama.org/) for the RL environment interface
- Inspired by the Pokemon Pinball game developed by Jupiter and published by Nintendo
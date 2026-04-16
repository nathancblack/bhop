"""Gymnasium environment wrapping Quake III movement physics.

Rewards horizontal speed to encourage discovery of bunnyhopping.
See docs/environment_design.md for full design details.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from bhop.geometry import MapGeometry
from bhop.physics import Q3Physics


class BhopEnv(gym.Env):
    """Flat-plane bunnyhopping environment.

    Observation: Box(5,) float32 -- [vel_x, vel_y, speed, vel_z, on_ground]
    Action: Box(4,) float32 -- [forward, right, jump, yaw_delta]
      - forward/right: thresholded at -0.33/+0.33 → {-127, 0, +127}
      - jump: thresholded at 0.0 → no jump / jump
      - yaw_delta: continuous degrees per tick in [-MAX_YAW_DEG, +MAX_YAW_DEG]
    Reward: horizontal_speed / 320.0 per tick
    Episode: 1000 ticks (8s at 125fps), start from zero velocity on ground.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    # Max yaw delta per tick in degrees (maps to [-MAX_YAW_DEG, +MAX_YAW_DEG])
    MAX_YAW_DEG: float = 5.0

    def __init__(
        self,
        render_mode: str | None = None,
        map_geometry: MapGeometry | None = None,
    ) -> None:
        """Initialize environment with observation and action spaces.

        Args:
            render_mode: Not used (no rendering support).
            map_geometry: Optional map geometry for collision detection.
                When None, uses flat infinite plane (original behavior).
        """
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=np.array([-2000, -2000, 0, -2000, 0], dtype=np.float32),
            high=np.array([2000, 2000, 2000, 2000, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -self.MAX_YAW_DEG], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, self.MAX_YAW_DEG], dtype=np.float32),
            dtype=np.float32,
        )

        self._physics = Q3Physics(geometry=map_geometry)
        self._max_speed: float = 0.0
        self._jump_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (initial observation, info dict).
        """
        super().reset(seed=seed, options=options)
        self._physics.reset()
        self._max_speed = 0.0
        self._jump_count = 0
        return self._get_obs(), self._get_info()

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Maps continuous action to physics inputs, ticks physics, computes reward.

        Args:
            action: Continuous action array [forward, right, jump, yaw_delta].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        forward_move, right_move, jump, yaw_delta = self._map_action(action)
        was_on_ground = self._physics.on_ground
        self._physics.tick(forward_move, right_move, jump, yaw_delta)

        speed = self._physics.horizontal_speed
        self._max_speed = max(self._max_speed, speed)
        if was_on_ground and not self._physics.on_ground:
            self._jump_count += 1

        reward = speed / 320.0
        return self._get_obs(), reward, False, False, self._get_info()

    def _get_obs(self) -> NDArray[np.float32]:
        """Build observation array from physics state.

        Returns:
            Float32 array: [vel_x, vel_y, speed, vel_z, on_ground]
        """
        state = self._physics.get_state()
        obs = np.array(
            [
                state["vel_x"],
                state["vel_y"],
                state["speed"],
                state["vel_z"],
                1.0 if state["on_ground"] else 0.0,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self) -> dict[str, Any]:
        """Build info dict with episode statistics.

        Returns:
            Dict with keys: speed, max_speed, jumps
        """
        return {
            "speed": self._physics.horizontal_speed,
            "max_speed": self._max_speed,
            "jumps": self._jump_count,
        }

    def _map_action(
        self, action: NDArray[np.float32]
    ) -> tuple[float, float, bool, float]:
        """Map continuous action to physics inputs.

        Thresholds:
            [0] forward: <-0.33 → backward(-127), >+0.33 → forward(+127), else 0
            [1] right:   <-0.33 → left(-127),     >+0.33 → right(+127),   else 0
            [2] jump:    >0.0 → jump
            [3] yaw_delta: continuous degrees, clipped to [-MAX_YAW_DEG, +MAX_YAW_DEG]

        Args:
            action: Raw continuous action array.

        Returns:
            Tuple of (forward_move, right_move, jump, yaw_delta_radians).
        """
        # Forward: threshold at ±0.33
        raw_fwd = float(action[0])
        if raw_fwd > 0.33:
            forward_move = 127.0
        elif raw_fwd < -0.33:
            forward_move = -127.0
        else:
            forward_move = 0.0

        # Right: threshold at ±0.33
        raw_right = float(action[1])
        if raw_right > 0.33:
            right_move = 127.0
        elif raw_right < -0.33:
            right_move = -127.0
        else:
            right_move = 0.0

        # Jump: threshold at 0.0
        jump = float(action[2]) > 0.0

        # Yaw delta: continuous degrees, clip to bounds, convert to radians
        yaw_deg = float(np.clip(action[3], -self.MAX_YAW_DEG, self.MAX_YAW_DEG))
        yaw_delta = np.radians(yaw_deg)

        return forward_move, right_move, jump, float(yaw_delta)

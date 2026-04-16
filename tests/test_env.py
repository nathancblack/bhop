"""Gymnasium environment compliance and correctness tests.

Organized by issue breakdown (docs/issues.md):
  - 3b: Registration and API compliance
  - 3c: Edge cases and info dict
"""

import gymnasium as gym
import numpy as np
import pytest

import bhop  # noqa: F401 -- triggers env registration
from bhop.env import BhopEnv


# ---------------------------------------------------------------------------
# Issue 3b: Environment registration and compliance
# ---------------------------------------------------------------------------


class TestEnvCompliance:
    """Gymnasium API compliance tests."""

    def test_env_creation(self) -> None:
        """Can create env via gymnasium.make('bhop/BhopFlat-v0')."""
        env = gym.make("bhop/BhopFlat-v0")
        assert env is not None
        assert isinstance(env.unwrapped, BhopEnv)
        env.close()

    def test_reset_returns_correct_types(self) -> None:
        """reset() returns (obs, info) with correct shapes and types."""
        env = BhopEnv()
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self) -> None:
        """step() returns (obs, reward, terminated, truncated, info) with correct types."""
        env = BhopEnv()
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_within_bounds(self) -> None:
        """Observation values stay within declared space bounds."""
        env = BhopEnv()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)

    def test_random_actions_full_episode(self) -> None:
        """Random actions don't crash over a full 1000-step episode."""
        env = BhopEnv()
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert not terminated  # episodes never terminate early

    def test_check_env(self) -> None:
        """SB3's check_env() passes."""
        from stable_baselines3.common.env_checker import check_env

        env = BhopEnv()
        check_env(env, warn=True)


# ---------------------------------------------------------------------------
# Issue 3c: Edge cases and info dict
# ---------------------------------------------------------------------------


class TestEnvEdgeCases:
    """Edge case handling and info dict tests."""

    def test_zero_velocity_yaw(self) -> None:
        """Zero velocity doesn't cause yaw computation errors."""
        env = BhopEnv()
        env.reset()
        # Take steps with various yaw values at zero velocity (no forward/right)
        for yaw_deg in np.linspace(-env.MAX_YAW_DEG, env.MAX_YAW_DEG, 11):
            action = np.array([0.0, 0.0, -1.0, yaw_deg], dtype=np.float32)
            obs, reward, _, _, _ = env.step(action)
            assert not np.any(np.isnan(obs))
            assert not np.any(np.isinf(obs))

    def test_info_dict_keys(self) -> None:
        """Info dict contains expected keys: speed, max_speed, jumps."""
        env = BhopEnv()
        _, info = env.reset()
        assert "speed" in info
        assert "max_speed" in info
        assert "jumps" in info
        # After reset, all should be zero
        assert info["speed"] == 0.0
        assert info["max_speed"] == 0.0
        assert info["jumps"] == 0

        # After a step with forward input, speed should be positive
        action = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)  # forward, no strafe, no jump, zero yaw
        _, _, _, _, info = env.step(action)
        assert info["speed"] > 0.0
        assert info["max_speed"] > 0.0
        assert info["jumps"] == 0

    def test_observation_bounds_respected(self) -> None:
        """Observations are clipped/bounded to declared space limits."""
        env = BhopEnv()
        env.reset()
        # Drive physics to high speed via direct manipulation, then check clipping
        env._physics.velocity[0] = 3000.0  # exceeds 2000 bound
        obs = env._get_obs()
        assert obs[0] == 2000.0  # vel_x clipped
        assert obs[2] == 2000.0  # speed clipped
        assert env.observation_space.contains(obs)

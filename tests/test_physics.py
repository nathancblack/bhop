"""Physics correctness tests for Q3Physics.

These are the most important tests in the project. If the physics are wrong,
everything downstream is meaningless.

Organized by issue breakdown (docs/issues.md):
  - 2a: Basic physics tests
  - 2b: PM_Accelerate exact value tests
  - 2c: Bhop tests (most important)
"""

import numpy as np
import pytest

from bhop.physics import Q3Physics


# ---------------------------------------------------------------------------
# Issue 2a: Basic physics tests
# ---------------------------------------------------------------------------


class TestBasicPhysics:
    """Basic physics behavior tests."""

    def test_standing_still_no_drift(self) -> None:
        """No inputs -> no movement."""
        phys = Q3Physics()
        for _ in range(100):
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
        np.testing.assert_array_equal(phys.velocity, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(phys.position, [0.0, 0.0, 0.0])

    def test_ground_acceleration_caps(self) -> None:
        """Forward held -> converges to ~320 ups."""
        phys = Q3Physics()
        for _ in range(1000):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
        assert 319 < phys.horizontal_speed < 321

    def test_friction_stops_player(self) -> None:
        """Moving player with no input -> decelerates to 0."""
        phys = Q3Physics()
        phys.velocity[0] = 300.0
        for _ in range(1000):
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
        assert phys.horizontal_speed < 1.0

    def test_cmdscale_diagonal(self) -> None:
        """Forward+right not faster than forward alone on ground."""
        phys = Q3Physics()
        for _ in range(1000):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
        forward_speed = phys.horizontal_speed

        phys.reset()
        for _ in range(1000):
            phys.tick(forward_move=127, right_move=127, jump=False, yaw_delta=0)
        diagonal_speed = phys.horizontal_speed

        assert abs(forward_speed - diagonal_speed) < 1.0

    def test_jump_velocity(self) -> None:
        """Jump sets vel_z ~= 270."""
        phys = Q3Physics()
        phys.tick(forward_move=0, right_move=0, jump=True, yaw_delta=0)
        assert 260 < phys.velocity[2] < 270

    def test_gravity_landing(self) -> None:
        """Player lands after jumping with no input."""
        phys = Q3Physics()
        phys.tick(forward_move=0, right_move=0, jump=True, yaw_delta=0)
        landed = False
        for _ in range(200):
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
            if phys.on_ground:
                landed = True
                break
        assert landed
        assert phys.position[2] == 0.0
        assert phys.velocity[2] == 0.0

    def test_no_air_friction(self) -> None:
        """Airborne player doesn't lose horizontal speed."""
        phys = Q3Physics()
        phys.velocity[0] = 500.0
        phys.on_ground = False
        phys.position[2] = 100.0
        for _ in range(50):
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
        assert phys.velocity[0] == 500.0

    def test_air_acceleration_weak(self) -> None:
        """Air acceleration is much slower than ground acceleration."""
        phys = Q3Physics()
        for _ in range(10):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
        ground_speed = phys.horizontal_speed

        phys.reset()
        phys.position[2] = 100.0
        phys.on_ground = False
        for _ in range(10):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
        air_speed = phys.horizontal_speed

        assert ground_speed > air_speed * 5


# ---------------------------------------------------------------------------
# Issue 2b: PM_Accelerate exact value tests
# ---------------------------------------------------------------------------


class TestPMAccelerate:
    """Exact numerical tests for the PM_Accelerate function."""

    def test_pm_accelerate_exact_ground(self) -> None:
        """Known velocity + wishdir -> exact expected result."""
        phys = Q3Physics()
        phys.velocity[:] = [0.0, 0.0, 0.0]
        phys._pm_accelerate(np.array([1.0, 0.0, 0.0]), 320.0, 10.0)
        np.testing.assert_allclose(phys.velocity, [25.6, 0.0, 0.0], atol=1e-10)

    def test_pm_accelerate_addspeed_clamp(self) -> None:
        """Verify addspeed clamping when near maxspeed."""
        phys = Q3Physics()
        phys.velocity[:] = [319.0, 0.0, 0.0]
        phys._pm_accelerate(np.array([1.0, 0.0, 0.0]), 320.0, 10.0)
        np.testing.assert_allclose(phys.velocity, [320.0, 0.0, 0.0], atol=1e-10)

    def test_pm_accelerate_no_accel_when_above(self) -> None:
        """No acceleration when projection exceeds wishspeed."""
        phys = Q3Physics()
        phys.velocity[:] = [500.0, 0.0, 0.0]
        phys._pm_accelerate(np.array([1.0, 0.0, 0.0]), 320.0, 10.0)
        np.testing.assert_allclose(phys.velocity, [500.0, 0.0, 0.0], atol=1e-10)

    def test_pm_accelerate_perpendicular(self) -> None:
        """Perpendicular wishdir gives maximum addspeed."""
        phys = Q3Physics()
        phys.velocity[:] = [500.0, 0.0, 0.0]
        phys._pm_accelerate(np.array([0.0, 1.0, 0.0]), 320.0, 1.0)
        np.testing.assert_allclose(phys.velocity[0], 500.0, atol=1e-10)
        np.testing.assert_allclose(phys.velocity[1], 2.56, atol=1e-10)
        assert np.sqrt(phys.velocity[0] ** 2 + phys.velocity[1] ** 2) > 500.0


# ---------------------------------------------------------------------------
# Issue 2c: The bhop test (most important test in the project)
# ---------------------------------------------------------------------------


class TestBhop:
    """Tests confirming the bunnyhopping exploit exists in the physics."""

    def test_bhop_exceeds_maxspeed(self) -> None:
        """Scripted bhop inputs produce speed > 400 ups."""
        phys = Q3Physics()
        yaw_rate = np.radians(0.4)

        # Build ground speed
        for _ in range(500):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)

        # Bhop: 15 cycles of release-jump-airstrafe
        for _ in range(15):
            # Release jump for one tick
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
            # Jump and airstrafe until landing
            phys.tick(forward_move=0, right_move=127, jump=True, yaw_delta=yaw_rate)
            for _ in range(200):
                if phys.on_ground:
                    break
                phys.tick(forward_move=0, right_move=127, jump=False, yaw_delta=yaw_rate)

        assert phys.horizontal_speed > 400

    def test_bhop_speed_increases_over_jumps(self) -> None:
        """Speed after jump N+1 > speed after jump N."""
        phys = Q3Physics()
        yaw_rate = np.radians(0.4)

        # Build ground speed
        for _ in range(500):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)

        # Run 5 bhop cycles, record speed at each landing
        speeds = []
        for _ in range(5):
            # Realign yaw to velocity direction (like a real bhopper adjusting aim)
            phys.yaw = np.arctan2(phys.velocity[1], phys.velocity[0])
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
            phys.tick(forward_move=0, right_move=127, jump=True, yaw_delta=yaw_rate)
            for _ in range(200):
                if phys.on_ground:
                    break
                phys.tick(forward_move=0, right_move=127, jump=False, yaw_delta=yaw_rate)
            speeds.append(phys.horizontal_speed)

        # Each landing should be faster than the previous
        for i in range(1, len(speeds)):
            assert speeds[i] > speeds[i - 1]

    def test_no_bhop_without_strafe(self) -> None:
        """Jumping without strafing does NOT exceed 320."""
        phys = Q3Physics()

        # Build ground speed
        for _ in range(500):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)

        # Bhop 15 cycles: jumping but no strafing (no right_move, no yaw rotation)
        for _ in range(15):
            # Release jump for one tick
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
            # Jump
            phys.tick(forward_move=127, right_move=0, jump=True, yaw_delta=0)
            # Air ticks until landing
            for _ in range(200):
                if phys.on_ground:
                    break
                phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)

        assert phys.horizontal_speed < 330

    def test_no_bhop_without_jumping(self) -> None:
        """Strafing on ground does NOT exceed 320."""
        phys = Q3Physics()

        # Strafe on ground with yaw rotation but never jump
        for _ in range(1000):
            phys.tick(
                forward_move=127,
                right_move=127,
                jump=False,
                yaw_delta=np.radians(0.4),
            )

        assert phys.horizontal_speed < 330

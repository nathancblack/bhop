"""Geometry and collision detection tests.

Tests trace functions, slide-move integration, ground detection on brushes,
and full physics compatibility with and without geometry.
"""

import numpy as np
import pytest

from bhop.geometry import (
    Brush,
    MapGeometry,
    TraceResult,
    corridor_map,
    platform_map,
    trace,
    trace_ray,
    turn_map,
)
from bhop.physics import Q3Physics


# ---------------------------------------------------------------------------
# Trace function tests
# ---------------------------------------------------------------------------


class TestTrace:
    """Tests for trace_ray and trace functions."""

    def test_trace_empty_space(self) -> None:
        """Trace through empty space returns fraction=1.0, no hit."""
        brush = Brush(mins=np.array([10, 10, 10]), maxs=np.array([20, 20, 20]))
        result = trace_ray(np.array([0, 0, 0.0]), np.array([5, 0, 0.0]), brush)
        assert not result.hit
        assert result.fraction == 1.0
        assert result.normal is None

    def test_trace_hit_wall_face_on(self) -> None:
        """Trace into a wall face-on returns correct fraction and normal."""
        brush = Brush(mins=np.array([5, -1, -1.0]), maxs=np.array([6, 1, 1.0]))
        result = trace_ray(np.array([0, 0, 0.0]), np.array([10, 0, 0.0]), brush)
        assert result.hit
        assert abs(result.fraction - 0.5) < 1e-10
        np.testing.assert_array_equal(result.normal, [-1, 0, 0])

    def test_trace_hit_wall_from_other_side(self) -> None:
        """Trace hitting wall from -X side has +X normal."""
        brush = Brush(mins=np.array([5, -1, -1.0]), maxs=np.array([6, 1, 1.0]))
        result = trace_ray(np.array([10, 0, 0.0]), np.array([0, 0, 0.0]), brush)
        assert result.hit
        np.testing.assert_array_equal(result.normal, [1, 0, 0])

    def test_trace_parallel_miss(self) -> None:
        """Trace parallel to slab, outside, returns no hit."""
        brush = Brush(mins=np.array([0, 0, 0.0]), maxs=np.array([10, 10, 10.0]))
        result = trace_ray(np.array([5, -1, 5.0]), np.array([5, -1, 15.0]), brush)
        assert not result.hit

    def test_trace_start_inside_brush(self) -> None:
        """Trace starting well inside brush returns allsolid (fraction=0)."""
        brush = Brush(mins=np.array([0, 0, 0.0]), maxs=np.array([10, 10, 10.0]))
        result = trace_ray(np.array([5, 5, 5.0]), np.array([15, 5, 5.0]), brush)
        assert result.hit
        assert result.fraction == 0.0

    def test_trace_on_surface_not_allsolid(self) -> None:
        """Trace starting on brush surface is NOT treated as allsolid."""
        brush = Brush(mins=np.array([0, -50, -64.0]), maxs=np.array([100, 50, 0.0]))
        # Player on top surface (z=0), moving horizontally
        result = trace_ray(np.array([50, 0, 0.0]), np.array([60, 0, 0.0]), brush)
        assert not result.hit, "On-surface horizontal trace should not be allsolid"

    def test_trace_too_short(self) -> None:
        """Trace too short to reach brush returns no hit."""
        brush = Brush(mins=np.array([20, -1, -1.0]), maxs=np.array([21, 1, 1.0]))
        result = trace_ray(np.array([0, 0, 0.0]), np.array([10, 0, 0.0]), brush)
        assert not result.hit

    def test_trace_multi_brush_nearest(self) -> None:
        """Multi-brush trace returns the nearest hit."""
        geo = MapGeometry()
        geo.add_brush((5, -1, -1), (6, 1, 1))
        geo.add_brush((8, -1, -1), (9, 1, 1))
        result = trace(np.array([0, 0, 0.0]), np.array([10, 0, 0.0]), geo)
        assert result.hit
        assert abs(result.fraction - 0.5) < 1e-10

    def test_trace_no_brushes(self) -> None:
        """Trace against empty geometry returns no hit."""
        geo = MapGeometry()
        result = trace(np.array([0, 0, 0.0]), np.array([10, 0, 0.0]), geo)
        assert not result.hit
        assert result.fraction == 1.0


# ---------------------------------------------------------------------------
# Collision integration tests (slide-move)
# ---------------------------------------------------------------------------


class TestCollision:
    """Tests for physics collision via slide-move."""

    def test_player_stops_at_wall(self) -> None:
        """Player running into a wall doesn't pass through."""
        geo = MapGeometry()
        geo.add_brush((100, -50, -10), (110, 50, 50))

        phys = Q3Physics(geometry=geo)
        phys.velocity[0] = 500.0
        for _ in range(200):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
        assert phys.position[0] < 101, f"Passed through wall: pos_x={phys.position[0]}"

    def test_player_slides_along_wall(self) -> None:
        """Player approaching wall at angle slides along it."""
        geo = MapGeometry()
        geo.add_brush((5, -500, -10), (6, 500, 50))

        phys = Q3Physics(geometry=geo)
        phys.velocity[0] = 300.0
        phys.velocity[1] = 300.0
        phys.position[2] = 50.0
        phys.on_ground = False

        for _ in range(10):
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)

        assert phys.position[0] < 6, "Should not pass through wall"
        assert phys.position[1] > 10, "Should have slid along wall in Y"
        assert abs(phys.velocity[0]) < 5, "X velocity should be near zero"
        assert phys.velocity[1] > 250, "Y velocity should be preserved"

    def test_corridor_walls_contain_player(self) -> None:
        """Player stays within corridor walls."""
        geo = corridor_map()
        phys = Q3Physics(geometry=geo)

        # Run diagonally to hit a wall
        phys.velocity[0] = 300.0
        phys.velocity[1] = 300.0
        for _ in range(500):
            phys.tick(forward_move=127, right_move=127, jump=False, yaw_delta=0)
        assert abs(phys.position[1]) <= 201, f"Escaped corridor: pos_y={phys.position[1]}"


# ---------------------------------------------------------------------------
# Ground trace with geometry tests
# ---------------------------------------------------------------------------


class TestGroundTrace:
    """Tests for ground detection on brush surfaces."""

    def test_stand_on_floor_brush(self) -> None:
        """Player stands on corridor floor brush (z=0 surface)."""
        geo = corridor_map()
        phys = Q3Physics(geometry=geo)
        phys.position[:] = [100, 0, 0.1]
        phys.velocity[2] = -5.0
        phys.on_ground = False

        phys._pm_ground_trace()
        assert phys.on_ground
        assert phys.velocity[2] == 0.0

    def test_stand_on_raised_platform(self) -> None:
        """Player stands on a raised platform (z=64 surface)."""
        geo = platform_map()
        phys = Q3Physics(geometry=geo)
        phys.position[:] = [300, 0, 64.1]
        phys.velocity[2] = -5.0
        phys.on_ground = False

        phys._pm_ground_trace()
        assert phys.on_ground
        assert abs(phys.position[2] - 64.0) < 0.5

    def test_no_ground_in_midair(self) -> None:
        """Player high above geometry is not on ground."""
        geo = corridor_map()
        phys = Q3Physics(geometry=geo)
        phys.position[:] = [100, 0, 50.0]
        phys.velocity[2] = -10.0
        phys.on_ground = False

        phys._pm_ground_trace()
        assert not phys.on_ground

    def test_world_floor_fallback(self) -> None:
        """Player at z=0 outside geometry still lands on world floor."""
        geo = MapGeometry()
        geo.add_brush((1000, -10, -10), (1010, 10, 10))  # far away brush

        phys = Q3Physics(geometry=geo)
        phys.position[:] = [0, 0, 0.0]
        phys.velocity[2] = -10.0
        phys.on_ground = False

        phys._pm_ground_trace()
        assert phys.on_ground

    def test_wall_face_not_ground(self) -> None:
        """Vertical wall face (normal_z=0) does not count as ground."""
        geo = MapGeometry()
        geo.add_brush((49, -50, -10), (51, 50, 100))

        phys = Q3Physics(geometry=geo)
        phys.position[:] = [48, 0, 50.0]
        phys.velocity[2] = -10.0
        phys.on_ground = False

        phys._pm_ground_trace()
        assert not phys.on_ground


# ---------------------------------------------------------------------------
# Full physics loop with geometry
# ---------------------------------------------------------------------------


class TestPhysicsWithGeometry:
    """Integration tests for full physics loop with geometry."""

    def test_bhop_in_corridor(self) -> None:
        """Player can bhop in a corridor and gain speed."""
        geo = corridor_map()
        phys = Q3Physics(geometry=geo)

        # Build ground speed
        for _ in range(200):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
        assert phys.horizontal_speed > 310

        # Bhop 5 cycles
        for _ in range(5):
            phys.yaw = np.arctan2(phys.velocity[1], phys.velocity[0])
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
            phys.tick(forward_move=0, right_move=127, jump=True, yaw_delta=np.radians(0.4))
            for _ in range(200):
                if phys.on_ground:
                    break
                phys.tick(forward_move=0, right_move=127, jump=False, yaw_delta=np.radians(0.4))

        assert phys.horizontal_speed > 400, f"Bhop should exceed 400, got {phys.horizontal_speed}"

    def test_no_geometry_backwards_compat(self) -> None:
        """Q3Physics() with no geometry is identical to original behavior."""
        phys_geo = Q3Physics(geometry=None)
        phys_orig = Q3Physics()

        phys_geo.velocity[0] = 300.0
        phys_orig.velocity[0] = 300.0

        for _ in range(100):
            phys_geo.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
            phys_orig.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)

        np.testing.assert_allclose(phys_geo.position, phys_orig.position, atol=1e-10)
        np.testing.assert_allclose(phys_geo.velocity, phys_orig.velocity, atol=1e-10)

    def test_jump_and_land_on_geometry(self) -> None:
        """Player jumps and lands correctly on floor brush."""
        geo = corridor_map()
        phys = Q3Physics(geometry=geo)

        # Jump
        phys.tick(forward_move=0, right_move=0, jump=True, yaw_delta=0)
        assert not phys.on_ground

        # Wait to land
        landed = False
        for _ in range(200):
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
            if phys.on_ground:
                landed = True
                break

        assert landed
        assert abs(phys.position[2]) < 0.1


# ---------------------------------------------------------------------------
# Original physics tests still pass (regression guard)
# ---------------------------------------------------------------------------


class TestOriginalPhysicsRegression:
    """Verify that all original flat-plane behaviors are unchanged."""

    def test_standing_still_no_drift(self) -> None:
        phys = Q3Physics()
        for _ in range(100):
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
        np.testing.assert_array_equal(phys.velocity, [0.0, 0.0, 0.0])

    def test_ground_acceleration_caps(self) -> None:
        phys = Q3Physics()
        for _ in range(1000):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
        assert 319 < phys.horizontal_speed < 321

    def test_bhop_exceeds_maxspeed(self) -> None:
        phys = Q3Physics()
        yaw_rate = np.radians(0.4)
        for _ in range(500):
            phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
        for _ in range(15):
            phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
            phys.tick(forward_move=0, right_move=127, jump=True, yaw_delta=yaw_rate)
            for _ in range(200):
                if phys.on_ground:
                    break
                phys.tick(forward_move=0, right_move=127, jump=False, yaw_delta=yaw_rate)
        assert phys.horizontal_speed > 400

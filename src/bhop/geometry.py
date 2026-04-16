"""Map geometry and collision detection for Quake III physics.

Provides AABB brush-based geometry with ray tracing for collision detection.
Designed to be dual-purpose: usable in Python training AND exportable to Q3 .map format.
See docs/quake3_physics.md for physics reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TraceResult:
    """Result of a ray trace against geometry.

    Attributes:
        fraction: How far along the ray the hit occurred (0.0 to 1.0).
            1.0 means no collision (ray traveled its full length).
        normal: Surface normal at the hit point, or None if no hit.
        hit: Whether the trace hit anything.
    """

    fraction: float
    normal: NDArray[np.float64] | None
    hit: bool


@dataclass
class Brush:
    """Axis-aligned bounding box (AABB) brush.

    Represents a solid volume. Maps directly to a Q3 brush primitive
    (6 axis-aligned planes) for .map export.

    Attributes:
        mins: Minimum corner (x, y, z).
        maxs: Maximum corner (x, y, z).
    """

    mins: NDArray[np.float64]
    maxs: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.mins = np.asarray(self.mins, dtype=np.float64)
        self.maxs = np.asarray(self.maxs, dtype=np.float64)


class MapGeometry:
    """Collection of AABB brushes forming a map.

    Attributes:
        brushes: List of solid Brush volumes.
    """

    def __init__(self) -> None:
        self.brushes: list[Brush] = []

    def add_brush(
        self,
        mins: tuple[float, float, float] | NDArray[np.float64],
        maxs: tuple[float, float, float] | NDArray[np.float64],
    ) -> Brush:
        """Add an AABB brush to the map.

        Args:
            mins: Minimum corner (x, y, z).
            maxs: Maximum corner (x, y, z).

        Returns:
            The created Brush.
        """
        brush = Brush(
            mins=np.asarray(mins, dtype=np.float64),
            maxs=np.asarray(maxs, dtype=np.float64),
        )
        self.brushes.append(brush)
        return brush


_DIST_EPSILON: float = 0.03125  # 1/32 unit, matches Q3's DIST_EPSILON


def trace_ray(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    brush: Brush,
) -> TraceResult:
    """Trace a ray against a single AABB brush using the slab method.

    Uses DIST_EPSILON (1/32 unit) for the parallel-axis containment check,
    matching Q3's approach: a point within epsilon of a brush face is
    considered outside, not inside. This prevents "allsolid" false positives
    when the player is standing on a brush surface.

    Args:
        start: Ray origin (3,).
        end: Ray endpoint (3,).
        brush: AABB brush to test against.

    Returns:
        TraceResult with fraction, normal, and hit flag.
    """
    direction = end - start
    t_enter = -np.inf
    t_exit = np.inf
    enter_axis = -1
    enter_sign = 0

    for i in range(3):
        if abs(direction[i]) < 1e-10:
            # Ray is parallel to slab planes on this axis.
            # Use epsilon-shrunk bounds: a point within DIST_EPSILON of a
            # face is considered outside (matches Q3 surface handling).
            if start[i] < brush.mins[i] + _DIST_EPSILON or start[i] > brush.maxs[i] - _DIST_EPSILON:
                return TraceResult(fraction=1.0, normal=None, hit=False)
        else:
            inv_d = 1.0 / direction[i]
            t1 = (brush.mins[i] - start[i]) * inv_d
            t2 = (brush.maxs[i] - start[i]) * inv_d

            if t1 > t2:
                # Ray enters through max face, exits through min face
                if t2 > t_enter:
                    t_enter = t2
                    enter_axis = i
                    enter_sign = 1  # normal points toward +axis (maxs face)
                t_exit = min(t_exit, t1)
            else:
                # Ray enters through min face, exits through max face
                if t1 > t_enter:
                    t_enter = t1
                    enter_axis = i
                    enter_sign = -1  # normal points toward -axis (mins face)
                t_exit = min(t_exit, t2)

            if t_enter > t_exit:
                return TraceResult(fraction=1.0, normal=None, hit=False)

    # Check if intersection is within the ray segment [0, 1]
    if t_enter < 0:
        # Ray starts inside or on the surface of the brush.
        # Only treat as allsolid if the exit is well past the start (not
        # just at the surface). A t_exit near 0 means the start point is
        # on the brush face, not truly inside.
        if t_exit > _DIST_EPSILON:
            return TraceResult(
                fraction=0.0,
                normal=np.zeros(3, dtype=np.float64),
                hit=True,
            )
        return TraceResult(fraction=1.0, normal=None, hit=False)

    if t_enter > 1.0:
        # Intersection is beyond the ray endpoint
        return TraceResult(fraction=1.0, normal=None, hit=False)

    # Build the surface normal (outward-facing from the brush)
    normal = np.zeros(3, dtype=np.float64)
    normal[enter_axis] = float(enter_sign)

    return TraceResult(fraction=float(t_enter), normal=normal, hit=True)


def trace(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    geometry: MapGeometry,
) -> TraceResult:
    """Trace a ray against all brushes in a MapGeometry, returning the earliest hit.

    Args:
        start: Ray origin (3,).
        end: Ray endpoint (3,).
        geometry: Map geometry containing brushes to test.

    Returns:
        TraceResult for the nearest hit, or fraction=1.0 if no hit.
    """
    best = TraceResult(fraction=1.0, normal=None, hit=False)

    for brush in geometry.brushes:
        result = trace_ray(start, end, brush)
        if result.hit and result.fraction < best.fraction:
            best = result

    return best


def corridor_map() -> MapGeometry:
    """Create a straight corridor map for testing.

    Layout (top-down, Y is left/right, X is forward):
        - Floor: z = -1 to 0, spanning full corridor
        - Left wall: y = 200 to 216
        - Right wall: y = -216 to -200
        - Corridor runs from x = -100 to x = 2000
        - Width: 400 units (±200), matching typical Q3 corridor scale
        - No ceiling (open top)

    Player spawns at origin (0, 0, 0) on the floor surface.
    """
    geo = MapGeometry()

    # Floor: thick slab just below z=0
    geo.add_brush((-100, -216, -64), (2000, 216, 0))

    # Left wall (positive Y side) -- extends below floor for proper sealing
    geo.add_brush((-100, 200, -64), (2000, 216, 128))

    # Right wall (negative Y side) -- extends below floor for proper sealing
    geo.add_brush((-100, -216, -64), (2000, -200, 128))

    return geo


def turn_map() -> MapGeometry:
    """Create a corridor with a 90-degree right turn.

    Layout (top-down):
        Straight section: x = -100..800, y = -200..200
        Turn section:     x = 600..1000, y = -600..200
        (The turn connects at x=600..800 overlap)

    Player spawns at origin heading +X, must turn to -Y at the corner.
    """
    geo = MapGeometry()

    # Floor (covers both sections)
    geo.add_brush((-100, -600, -64), (1000, 216, 0))

    # Straight section walls -- extend below floor for sealing
    # Left wall (full length along +Y)
    geo.add_brush((-100, 200, -64), (1000, 216, 128))
    # Right wall (only up to the turn)
    geo.add_brush((-100, -216, -64), (600, -200, 128))

    # Turn section walls -- extend below floor for sealing
    # Far wall (end of straight, past the turn opening)
    geo.add_brush((800, -600, -64), (1000, 216, 128))
    # Outer wall of turn (along -Y)
    geo.add_brush((600, -616, -64), (1000, -600, 128))

    return geo


def platform_map() -> MapGeometry:
    """Create a map with a raised platform for ground trace testing.

    Layout:
        - Ground floor at z=0
        - Raised platform (z=0 to z=64) at x=200..400
        - Player spawns at origin on the ground floor
    """
    geo = MapGeometry()

    # Ground floor
    geo.add_brush((-100, -200, -64), (600, 200, 0))

    # Raised platform
    geo.add_brush((200, -100, 0), (400, 100, 64))

    return geo

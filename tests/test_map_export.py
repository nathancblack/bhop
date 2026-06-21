"""Tests for Q3 .map file exporter.

Covers:
  - Brush-to-planes conversion (6 planes per AABB)
  - Full .map file structure (worldspawn, brushes, spawn entity)
  - Integration with geometry.py map factories
"""

import os
import tempfile

import pytest

from bhop.geometry import Brush, MapGeometry, corridor_map, platform_map, turn_map
from bhop.map_export import _brush_to_planes, export_map


class TestBrushToPlanes:
    """Tests for AABB -> 6 half-plane conversion."""

    def test_unit_cube(self) -> None:
        """Unit cube at origin produces 6 planes."""
        brush = Brush(mins=(0, 0, 0), maxs=(1, 1, 1))
        planes = _brush_to_planes(brush)
        assert len(planes) == 6

    def test_planes_contain_texture(self) -> None:
        """Each plane line ends with texture name and params."""
        brush = Brush(mins=(0, 0, 0), maxs=(64, 64, 64))
        planes = _brush_to_planes(brush)
        for plane in planes:
            assert "common/caulk" in plane
            assert "0 0 0 1 1" in plane

    def test_planes_have_three_points(self) -> None:
        """Each plane has exactly 3 parenthesized point groups."""
        brush = Brush(mins=(-100, -200, -64), maxs=(2000, 200, 128))
        planes = _brush_to_planes(brush)
        for plane in planes:
            # Count ( x y z ) groups
            count = plane.count("(")
            assert count == 3, f"expected 3 points, got {count}: {plane}"

    def test_integer_coordinates(self) -> None:
        """Integer brush coordinates are written without decimals."""
        brush = Brush(mins=(0, 0, 0), maxs=(64, 128, 256))
        planes = _brush_to_planes(brush)
        joined = "\n".join(planes)
        # Should contain "64" not "64.0"
        assert "64.0" not in joined
        assert "128.0" not in joined

    def test_float_coordinates(self) -> None:
        """Non-integer brush coordinates are written with decimals."""
        brush = Brush(mins=(0, 0, 0), maxs=(64.5, 128, 256))
        planes = _brush_to_planes(brush)
        joined = "\n".join(planes)
        assert "64.5" in joined


class TestExportMap:
    """Tests for full .map file export."""

    def _export_and_read(
        self, geometry: MapGeometry, **kwargs
    ) -> str:
        """Export to a temp file and return contents."""
        with tempfile.NamedTemporaryFile(
            suffix=".map", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            export_map(geometry, path, **kwargs)
            with open(path) as f:
                return f.read()
        finally:
            os.unlink(path)

    def test_worldspawn_entity(self) -> None:
        """Output contains a worldspawn entity."""
        g = MapGeometry()
        g.add_brush(mins=(0, 0, -64), maxs=(64, 64, 0))
        content = self._export_and_read(g)
        assert '"classname" "worldspawn"' in content

    def test_spawn_entity(self) -> None:
        """Output contains an info_player_deathmatch entity."""
        g = MapGeometry()
        g.add_brush(mins=(0, 0, -64), maxs=(64, 64, 0))
        content = self._export_and_read(g)
        assert '"classname" "info_player_deathmatch"' in content

    def test_spawn_origin(self) -> None:
        """Spawn origin appears in the output."""
        g = MapGeometry()
        g.add_brush(mins=(0, 0, -64), maxs=(64, 64, 0))
        content = self._export_and_read(g, spawn_origin=(100, 200, 24))
        assert '"origin" "100 200 24"' in content

    def test_brush_count(self) -> None:
        """Number of brush blocks matches geometry brush count."""
        g = MapGeometry()
        g.add_brush(mins=(0, 0, -64), maxs=(100, 100, 0))
        g.add_brush(mins=(0, 90, -64), maxs=(100, 100, 128))
        content = self._export_and_read(g)
        # Count brush blocks: each brush has { ... 6 planes ... }
        # The worldspawn { } also has braces, and spawn entity too
        # Count "common/caulk" lines: 6 per brush
        caulk_count = content.count("common/caulk")
        assert caulk_count == 12  # 2 brushes * 6 planes

    def test_map_name_in_comment(self) -> None:
        """Map name appears in the header comment."""
        g = MapGeometry()
        g.add_brush(mins=(0, 0, -64), maxs=(64, 64, 0))
        content = self._export_and_read(g, map_name="test_map")
        assert "test_map" in content

    def test_corridor_map_export(self) -> None:
        """corridor_map() exports with 3 brushes."""
        g = corridor_map()
        content = self._export_and_read(g, map_name="bhop_corridor")
        caulk_count = content.count("common/caulk")
        assert caulk_count == 18  # 3 brushes * 6 planes

    def test_platform_map_export(self) -> None:
        """platform_map() exports without error."""
        g = platform_map()
        content = self._export_and_read(g)
        assert '"classname" "worldspawn"' in content
        assert len(content) > 100

    def test_turn_map_export(self) -> None:
        """turn_map() exports without error."""
        g = turn_map()
        content = self._export_and_read(g)
        assert '"classname" "worldspawn"' in content
        assert len(content) > 100

    def test_empty_geometry(self) -> None:
        """Empty geometry still produces valid worldspawn + spawn point."""
        g = MapGeometry()
        content = self._export_and_read(g)
        assert '"classname" "worldspawn"' in content
        assert '"classname" "info_player_deathmatch"' in content

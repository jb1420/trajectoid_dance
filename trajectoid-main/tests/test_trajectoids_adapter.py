import sys
import types
import unittest
from unittest import mock

import numpy as np

import trajectoids_adapter as ta


def _circle_points(n: int = 64, radius: float = 1.0) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


class AdapterUtilityTests(unittest.TestCase):
    def test_get_minimize_scalar_available_or_none(self) -> None:
        value = ta._get_minimize_scalar()
        self.assertTrue(value is None or callable(value))

    def test_path_length_edge_and_nominal(self) -> None:
        self.assertEqual(ta.path_length(np.array([[0.0, 0.0]])), 0.0)
        points = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]])
        self.assertAlmostEqual(ta.path_length(points), 9.0, places=8)

    def test_resample_uniform_open_closed_and_zero_length(self) -> None:
        segment = np.array([[0.0, 0.0], [10.0, 0.0]])
        sampled = ta.resample_uniform(segment, n_points=5, closed=False)
        self.assertEqual(sampled.shape, (5, 2))
        np.testing.assert_allclose(sampled[:, 0], [0.0, 2.5, 5.0, 7.5, 10.0], atol=1e-8)

        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        sampled_closed = ta.resample_uniform(square, n_points=8, closed=True)
        self.assertEqual(sampled_closed.shape, (8, 2))
        self.assertGreater(np.linalg.norm(sampled_closed[0] - sampled_closed[-1]), ta.EPS)

        repeated = np.array([[1.0, 2.0], [1.0, 2.0]])
        sampled_repeated = ta.resample_uniform(repeated, n_points=7, closed=False)
        np.testing.assert_allclose(sampled_repeated, repeated)

    def test_smooth_path_shapes_and_endpoints(self) -> None:
        open_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        unchanged = ta.smooth_path(open_points, passes=0, closed=False)
        np.testing.assert_allclose(unchanged, open_points)

        smoothed_open = ta.smooth_path(open_points, passes=1, closed=False)
        self.assertEqual(smoothed_open.shape, (6, 2))
        np.testing.assert_allclose(smoothed_open[0], open_points[0])
        np.testing.assert_allclose(smoothed_open[-1], open_points[-1])

        closed_points = np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=float,
        )
        smoothed_closed = ta.smooth_path(closed_points, passes=1, closed=True)
        self.assertEqual(smoothed_closed.shape, (8, 2))

    def test_curvature_profile_edge_and_line(self) -> None:
        s, kappa = ta.curvature_profile(np.array([[0.0, 0.0], [1.0, 0.0]]))
        self.assertEqual(s.size, 0)
        self.assertEqual(kappa.size, 0)

        line = np.column_stack([np.linspace(0.0, 10.0, 9), np.zeros(9)])
        s_line, kappa_line = ta.curvature_profile(line)
        self.assertEqual(s_line.shape[0], line.shape[0])
        self.assertEqual(kappa_line.shape[0], line.shape[0])
        self.assertLess(float(np.max(np.abs(kappa_line))), 1e-6)

    def test_validate_path_reports_expected_errors_and_suggestions(self) -> None:
        too_few = np.array([[0.0, 0.0], [1.0, 0.0]])
        val = ta.validate_path(too_few, require_closed=True)
        self.assertTrue(any("too few points" in msg for msg in val.errors))

        with_repeats = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [30.0, 0.0],
                [31.0, 0.0],
                [32.0, 0.0],
            ]
        )
        val_repeat = ta.validate_path(with_repeats, require_closed=True)
        self.assertTrue(any("zero-length segments" in msg for msg in val_repeat.errors))
        self.assertTrue(any("periodic/closed" in msg for msg in val_repeat.errors))
        self.assertTrue(any("uneven" in msg for msg in val_repeat.suggestions))

        zero_len = np.repeat([[5.0, 5.0]], repeats=8, axis=0)
        val_zero = ta.validate_path(zero_len, require_closed=False)
        self.assertTrue(any("effectively zero" in msg for msg in val_zero.errors))

    def test_rodrigues_rotation_matrix_and_angle(self) -> None:
        identity = ta._rodrigues_rotation_matrix(np.array([0.0, 0.0, 0.0]), 1.23)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-12)

        rot_z = ta._rodrigues_rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
        rotated_x = rot_z @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated_x, np.array([0.0, 1.0, 0.0]), atol=1e-7)

        self.assertAlmostEqual(ta._rotation_angle(np.eye(3)), 0.0, places=10)
        rot_pi_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=float,
        )
        self.assertAlmostEqual(ta._rotation_angle(rot_pi_x), np.pi, places=10)

    def test_rotation_from_point_to_point_and_rotations_to_origin(self) -> None:
        point = np.array([1.0, 0.0], dtype=float)
        previous = np.array([0.0, 0.0], dtype=float)
        rot = ta._rotation_from_point_to_point(point, previous)
        expected = ta._rodrigues_rotation_matrix(np.array([0.0, 1.0, 0.0]), -1.0)
        np.testing.assert_allclose(rot, expected, atol=1e-10)

        path = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.3]], dtype=float)
        rotations = ta.rotations_to_origin(path)
        self.assertEqual(rotations.shape, (3, 3, 3))
        np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-12)

    def test_trace_on_sphere_and_mismatch_angle(self) -> None:
        path = np.array([[0.0, 0.0], [0.3, 0.0], [0.3, 0.4]], dtype=float)
        trace = ta.trace_on_sphere(path, scale=1.0, core_radius=2.0)
        self.assertEqual(trace.shape, (3, 3))
        norms = np.linalg.norm(trace, axis=1)
        np.testing.assert_allclose(norms, np.full(3, 2.0), atol=1e-7)

        self.assertAlmostEqual(ta.mismatch_angle(np.array([[0.0, 0.0]]), scale=3.0), 0.0, places=10)
        self.assertGreaterEqual(ta.mismatch_angle(path, scale=1.0), 0.0)

    def test_objective_for_scale(self) -> None:
        path = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.2]], dtype=float)
        value = ta._objective_for_scale(path, scale=1.3, core_radius=1.0)
        angle = ta.mismatch_angle(path, 1.3)
        trace = ta.trace_on_sphere(path, 1.3, core_radius=1.0)
        gap = float(np.linalg.norm(trace[-1] - trace[0]))
        self.assertAlmostEqual(value, angle + 0.75 * gap, places=12)

    def test_estimate_scale_with_fallback_grid_search(self) -> None:
        path = _circle_points(n=48, radius=1.0)
        with mock.patch.object(ta, "_get_minimize_scalar", return_value=None):
            scale, angle, gap = ta.estimate_scale(path, core_radius=1.0)
        self.assertTrue(np.isfinite(scale) and scale > 0.0)
        self.assertTrue(np.isfinite(angle))
        self.assertTrue(np.isfinite(gap))

    def test_estimate_scale_with_optimizer_success_and_failure(self) -> None:
        path = _circle_points(n=32, radius=1.0)

        class _Result:
            def __init__(self, success: bool, x: float) -> None:
                self.success = success
                self.x = x

        def _minimize_success(_f, bounds, method, options):
            self.assertEqual(method, "bounded")
            self.assertEqual(bounds[0] < bounds[1], True)
            self.assertIn("xatol", options)
            return _Result(success=True, x=1.2345)

        with mock.patch.object(ta, "_get_minimize_scalar", return_value=_minimize_success):
            scale_success, _, _ = ta.estimate_scale(path, core_radius=1.0)
        self.assertAlmostEqual(scale_success, 1.2345, places=8)

        def _minimize_failure(_f, bounds, method, options):
            return _Result(success=False, x=9.9)

        with mock.patch.object(ta, "_get_minimize_scalar", return_value=_minimize_failure):
            scale_failure, _, _ = ta.estimate_scale(path, core_radius=1.0)
        expected_base = 2.0 * np.pi / max(ta.path_length(path), ta.EPS)
        self.assertAlmostEqual(scale_failure, expected_base, places=10)

    def test_compute_normals_normalization_and_decimation(self) -> None:
        path = _circle_points(n=120, radius=1.0)
        normals = ta._compute_normals(path, scale=1.0, core_radius=1.0, max_planes=20)
        self.assertEqual(normals.shape[1], 3)
        self.assertLessEqual(normals.shape[0], 20)
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-6)

    def test_implicit_field_small_grid(self) -> None:
        normals = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        lin, field = ta._implicit_field(
            normals=normals,
            outer_radius=1.2,
            core_radius=1.0,
            resolution=8,
            chunk_size=16,
            normal_batch=1,
        )
        self.assertEqual(lin.shape, (8,))
        self.assertEqual(field.shape, (8, 8, 8))
        self.assertTrue(np.all(np.isfinite(field)))

    def test_field_to_mesh_with_mocked_skimage_and_error_branch(self) -> None:
        fake_measure = types.SimpleNamespace(
            marching_cubes=lambda _field, level, spacing: (
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                np.array([[0, 1, 2]], dtype=np.int64),
                None,
                None,
            )
        )
        fake_skimage = types.SimpleNamespace(measure=fake_measure)
        lin = np.array([-2.0, 0.0, 2.0], dtype=float)
        field = np.zeros((3, 3, 3), dtype=float)

        with mock.patch.dict(sys.modules, {"skimage": fake_skimage}):
            verts, faces = ta._field_to_mesh(lin, field)
        self.assertEqual(verts.dtype, np.float32)
        self.assertEqual(faces.dtype, np.int32)
        np.testing.assert_allclose(verts[0], np.array([-2.0, -2.0, -2.0], dtype=np.float32))

        with mock.patch.dict(sys.modules, {"skimage": fake_skimage}):
            with self.assertRaisesRegex(ValueError, "No valid solid region"):
                ta._field_to_mesh(lin, np.ones((3, 3, 3), dtype=float))

    def test_clean_path_and_sample_open_polyline_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nx2"):
            ta._clean_path(np.array([0.0, 1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "too few points"):
            ta._clean_path(np.array([[0.0, 0.0]]))
        with self.assertRaisesRegex(ValueError, "no valid segments"):
            ta._clean_path(np.array([[0.0, 0.0], [0.0, 0.0]]))

        cleaned = ta._clean_path(np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0]]))
        np.testing.assert_allclose(cleaned, np.array([[0.0, 0.0], [2.0, 0.0]]))

        polyline = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=float)
        cumulative = np.array([0.0, 2.0, 4.0], dtype=float)
        sampled = ta._sample_open_polyline_batch(polyline, cumulative, np.array([-1.0, 99.0, 1.0, 3.0]))
        expected = np.array(
            [
                [0.0, 0.0],
                [2.0, 2.0],
                [1.0, 0.0],
                [2.0, 1.0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(sampled, expected, atol=1e-8)

    def test_build_roll_simulation_edge_cases_and_open_closed_behavior(self) -> None:
        base_path = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)

        with self.assertRaisesRegex(ValueError, "core_radius must be positive"):
            ta.build_roll_simulation(base_path, core_radius=0.0)

        sim_min_frames = ta.build_roll_simulation(
            base_path,
            target_roll_angle_rad=1.0,
            closed=False,
            n_frames=1,
            core_radius=1.0,
        )
        self.assertEqual(sim_min_frames.translations_xyz.shape[0], 2)

        sim_open_short = ta.build_roll_simulation(
            base_path,
            target_roll_angle_rad=10.0,
            closed=False,
            n_frames=12,
            core_radius=1.0,
        )
        self.assertFalse(sim_open_short.completed_target)
        self.assertIn("ended before", sim_open_short.message)

        closed_path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=float)
        sim_closed = ta.build_roll_simulation(
            closed_path,
            target_roll_angle_rad=6.5,
            closed=True,
            n_frames=16,
            core_radius=1.5,
        )
        self.assertTrue(sim_closed.completed_target)
        np.testing.assert_allclose(sim_closed.translations_xyz[:, 2], np.full(16, 1.5), atol=1e-10)

        sim_zero_target = ta.build_roll_simulation(
            base_path,
            target_roll_angle_rad=-5.0,
            closed=False,
            n_frames=6,
            core_radius=1.0,
        )
        self.assertAlmostEqual(sim_zero_target.achieved_roll_angle_rad, 0.0, places=10)
        self.assertTrue(np.allclose(sim_zero_target.trajectory_xy, sim_zero_target.trajectory_xy[0]))

    def test_generate_trajectoid_mesh_validation_instability_and_success(self) -> None:
        with self.assertRaisesRegex(ValueError, "too few points"):
            ta.generate_trajectoid_mesh(np.array([[0.0, 0.0], [1.0, 0.0]]), require_closed=False)

        path = _circle_points(n=40, radius=1.0)
        with mock.patch.object(ta, "estimate_scale", return_value=(1.0, np.deg2rad(80.0), 0.1)):
            with self.assertRaisesRegex(ValueError, "mismatch is too high"):
                ta.generate_trajectoid_mesh(path, require_closed=False)

        fake_vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        fake_faces = np.array([[0, 1, 2]], dtype=np.int32)
        fake_normals = np.array([[0.0, 0.0, -1.0]], dtype=float)
        fake_lin = np.array([-1.0, 1.0], dtype=float)
        fake_field = np.zeros((2, 2, 2), dtype=float)

        with mock.patch.object(ta, "estimate_scale", return_value=(1.1, 0.1, 0.02)), mock.patch.object(
            ta, "_compute_normals", return_value=fake_normals
        ), mock.patch.object(
            ta, "_implicit_field", return_value=(fake_lin, fake_field)
        ), mock.patch.object(
            ta, "_field_to_mesh", return_value=(fake_vertices, fake_faces)
        ):
            result = ta.generate_trajectoid_mesh(path, require_closed=False, resample_points=50)

        self.assertIsInstance(result, ta.GenerationResult)
        np.testing.assert_allclose(result.vertices, fake_vertices)
        np.testing.assert_allclose(result.faces, fake_faces)
        np.testing.assert_allclose(result.normals, fake_normals)
        self.assertAlmostEqual(result.scale, 1.1, places=10)
        self.assertEqual(result.resampled_points.shape[0], 50)


if __name__ == "__main__":
    unittest.main()

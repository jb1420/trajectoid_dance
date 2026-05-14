import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajectoids_adapter import export_binary_stl


class ExportBinaryStlTests(unittest.TestCase):
    def test_writes_expected_size_and_facet_count(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "quad.stl"
            written_path = export_binary_stl(vertices, faces, output_path)
            data = output_path.read_bytes()

        self.assertEqual(written_path, output_path.resolve())
        self.assertEqual(len(data), 84 + 50 * faces.shape[0])
        self.assertEqual(struct.unpack("<I", data[80:84])[0], faces.shape[0])

    def test_rejects_invalid_mesh_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "invalid.stl"

            with self.assertRaisesRegex(ValueError, "Nx3"):
                export_binary_stl(np.array([[0.0, 1.0]], dtype=float), np.array([[0, 0, 0]]), output_path)

            with self.assertRaisesRegex(TypeError, "integer-like"):
                export_binary_stl(
                    np.zeros((3, 3), dtype=float),
                    np.array([[0.1, 1.0, 2.0]], dtype=float),
                    output_path,
                )

            with self.assertRaisesRegex(ValueError, "outside the valid range"):
                export_binary_stl(
                    np.zeros((3, 3), dtype=float),
                    np.array([[0, 1, 7]], dtype=np.int32),
                    output_path,
                )

    def test_degenerate_triangles_write_zero_normals(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "degenerate.stl"
            export_binary_stl(vertices, faces, output_path)
            data = output_path.read_bytes()

        normal = np.array(struct.unpack("<3f", data[84:96]), dtype=float)
        self.assertEqual(len(data), 84 + 50)
        self.assertTrue(np.all(np.isfinite(normal)))
        self.assertTrue(np.allclose(normal, np.zeros(3)))


if __name__ == "__main__":
    unittest.main()

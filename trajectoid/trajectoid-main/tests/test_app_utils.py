import tempfile
import unittest
from pathlib import Path
from unittest import mock


try:
    import app as app_module
except Exception as exc:  # pragma: no cover - optional dependency environments
    app_module = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(app_module is None, f"app import unavailable: {_IMPORT_ERROR}")
class AppUtilityTests(unittest.TestCase):
    def test_hex_to_rgba_valid_and_invalid(self) -> None:
        rgba = app_module._hex_to_rgba("#ff8040", alpha=0.3)
        self.assertAlmostEqual(rgba[0], 1.0, places=10)
        self.assertAlmostEqual(rgba[1], 128.0 / 255.0, places=10)
        self.assertAlmostEqual(rgba[2], 64.0 / 255.0, places=10)
        self.assertAlmostEqual(rgba[3], 0.3, places=10)

        invalid = app_module._hex_to_rgba("abc", alpha=0.7)
        self.assertEqual(invalid, (0.0, 0.0, 0.0, 0.7))

    def test_is_writable_dir_true_and_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertTrue(app_module._is_writable_dir(Path(tmp_dir) / "nested"))
            file_path = Path(tmp_dir) / "not_a_dir"
            file_path.write_text("x", encoding="utf-8")
            self.assertFalse(app_module._is_writable_dir(file_path))

    def test_main_wires_qt_objects(self) -> None:
        fake_app = mock.Mock()
        fake_app.exec.return_value = 123
        fake_window = mock.Mock()

        with mock.patch.object(app_module.QtCore.QCoreApplication, "setAttribute") as set_attr, mock.patch.object(
            app_module.QtWidgets, "QApplication", return_value=fake_app
        ) as qapp_cls, mock.patch.object(app_module, "MainWindow", return_value=fake_window):
            result = app_module.main()

        self.assertEqual(result, 123)
        set_attr.assert_called()
        qapp_cls.assert_called_once()
        fake_window.show.assert_called_once()
        fake_app.exec.assert_called_once()


if __name__ == "__main__":
    unittest.main()

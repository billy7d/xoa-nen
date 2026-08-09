from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "sidecar"
if str(SIDECAR) not in sys.path:
    sys.path.insert(0, str(SIDECAR))

from cutout_sidecar.coordinator import Coordinator  # noqa: E402
from cutout_sidecar.image_core import decode_canonical, load_canonical_png  # noqa: E402
from cutout_sidecar.processor import artwork_alpha  # noqa: E402
from cutout_sidecar.project_store import ProjectStore  # noqa: E402
from cutout_sidecar.protocol import handle_request, serve_stdio  # noqa: E402


def artwork_fixture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    height, width = 96, 128
    rgb = np.full((height, width, 3), 248, dtype=np.uint8)
    rgb[18:78, 24:104] = (27, 65, 54)
    rgb[34:62, 45:83] = (230, 77, 65)
    rgb[8:13, 111:116] = (27, 65, 54)  # intentional detached component
    rgb[44:51, 59:66] = (248, 248, 248)  # negative-space hole
    alpha = np.full((height, width), 255, dtype=np.uint8)
    alpha[34:62, 45:83] = 180  # source-alpha constraint
    rgba = np.dstack((rgb, alpha))
    Image.fromarray(rgba, "RGBA").save(path)
    return rgb, alpha.astype(np.float32) / 255.0


class CanonicalImageTests(unittest.TestCase):
    def test_decode_preserves_rgba_and_source_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "tác phẩm 🐉.png"
            expected_rgb, expected_alpha = artwork_fixture(source)
            canonical = decode_canonical(source)
            self.assertEqual((canonical.width, canonical.height), (128, 96))
            np.testing.assert_array_equal(canonical.rgb, expected_rgb)
            np.testing.assert_allclose(canonical.source_alpha, expected_alpha, atol=0)
            self.assertEqual(canonical.original_orientation, 1)

    def test_exif_orientation_is_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "orientation-6.jpg"
            image = Image.new("RGB", (7, 3), (220, 20, 30))
            exif = image.getexif()
            exif[274] = 6
            image.save(source, exif=exif)
            canonical = decode_canonical(source)
            self.assertEqual(canonical.original_orientation, 6)
            self.assertEqual((canonical.width, canonical.height), (3, 7))

    def test_animated_webp_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "animated.webp"
            frames = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
            frames[0].save(source, save_all=True, append_images=frames[1:], duration=50, format="WEBP")
            with self.assertRaisesRegex(ValueError, "ảnh động"):
                decode_canonical(source)


class ArtworkEngineTests(unittest.TestCase):
    def test_background_is_removed_without_reanimating_source_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "art.png"
            rgb, source_alpha = artwork_fixture(source)
            result, diagnostics = artwork_alpha(rgb, source_alpha, tolerance=25, softness=10)
            self.assertLess(float(result[0, 0]), 0.01)
            self.assertGreater(float(result[22, 30]), 0.98)
            self.assertGreater(float(result[10, 113]), 0.98)
            self.assertLessEqual(float(np.max(result - source_alpha)), 1e-6)
            self.assertEqual(diagnostics["engine"], "classical-artwork-v1")


class CoordinatorFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source = root / "nested unicode" / "áo POD.png"
        self.source.parent.mkdir()
        self.expected_rgb, self.source_alpha = artwork_fixture(self.source)
        self.coordinator = Coordinator(ProjectStore(root / "projects"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_import_process_edit_undo_redo_preflight_and_exports(self) -> None:
        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        project_id = imported["project_id"]
        self.assertEqual(imported["schema_version"], "2.2.0")
        self.assertEqual(imported["canonical"]["decoded_pixel_hash"], imported["manifest"]["source"]["canonical_pixels_sha256"])
        self.assertTrue(Path(imported["preview_path"]).is_file())

        processed = self.coordinator.dispatch(
            "process_artwork",
            {"project_id": project_id, "tolerance": 25, "softness": 10, "quality_preset": "QUALITY"},
        )
        self.assertEqual(processed["process"]["content_mode"], "ARTWORK")
        self.assertGreaterEqual(processed["components"]["count"], 2)
        first_worker_pid = processed["process"]["diagnostics"]["worker_pid"]
        assert self.coordinator.worker.process is not None
        self.coordinator.worker.process.kill()
        self.coordinator.worker.process.wait(timeout=2)
        recovered = self.coordinator.dispatch(
            "process_artwork",
            {"project_id": project_id, "tolerance": 25, "softness": 10},
        )
        self.assertNotEqual(
            recovered["process"]["diagnostics"]["worker_pid"], first_worker_pid
        )
        before = self.coordinator.store.read_alpha(project_id)

        edited = self.coordinator.dispatch(
            "apply_brush",
            {
                "project_id": project_id,
                "points": [{"x": 31.5, "y": 24.5}, {"x": 36.5, "y": 24.5}],
                "radius": 5,
                "hardness": 1,
                "opacity": 1,
                "mode": "remove",
            },
        )
        after = self.coordinator.store.read_alpha(project_id)
        self.assertLess(float(after[24, 32]), float(before[24, 32]))
        self.assertTrue(edited["history"]["can_undo"])

        undone = self.coordinator.dispatch("undo", {"project_id": project_id})
        np.testing.assert_array_equal(self.coordinator.store.read_alpha(project_id), before)
        self.assertTrue(undone["history"]["can_redo"])
        redone = self.coordinator.dispatch("redo", {"project_id": project_id})
        np.testing.assert_array_equal(self.coordinator.store.read_alpha(project_id), after)
        self.assertTrue(redone["history"]["can_undo"])

        report = self.coordinator.dispatch(
            "preflight",
            {"project_id": project_id, "print_width_inch": 12, "print_height_inch": 10},
        )
        self.assertEqual(report["effective_ppi"]["x"], 128 / 12)
        self.assertEqual(report["status"], "WARN")
        self.assertTrue(Path(report["report_path"]).is_file())

        metric_report = self.coordinator.dispatch(
            "preflight",
            {
                "project_id": project_id,
                "print_width": 30.48,
                "print_height": 25.4,
                "print_unit": "cm",
            },
        )
        self.assertEqual(metric_report["print_dimensions"]["unit"], "cm")
        self.assertAlmostEqual(metric_report["print_dimensions"]["width_inch"], 12.0)
        self.assertAlmostEqual(metric_report["print_dimensions"]["height_inch"], 10.0)
        self.assertAlmostEqual(metric_report["effective_ppi"]["x"], 128 / 12)
        self.assertAlmostEqual(metric_report["effective_ppi"]["y"], 96 / 10)

        export_directory = Path(self.temporary.name) / "exports"
        master_path = export_directory / "master.png"
        master = self.coordinator.dispatch(
            "export",
            {"project_id": project_id, "output_mode": "MASTER_SOURCE_FAITHFUL", "destination": str(master_path)},
        )
        self.assertTrue(master["rgb_integrity"])
        with Image.open(master_path) as image:
            exported = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        np.testing.assert_array_equal(exported[:, :, :3], self.expected_rgb)

        pod_path = export_directory / "pod.png"
        pod = self.coordinator.dispatch(
            "export",
            {
                "project_id": project_id,
                "output_mode": "POD_READY",
                "destination": str(pod_path),
                "settings": {"trim": True, "padding": 3, "target_ppi": 300},
            },
        )
        self.assertLess(pod["width"], 128)
        with Image.open(pod_path) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertIsNotNone(image.info.get("icc_profile"))

        alpha_path = export_directory / "alpha.png"
        alpha_result = self.coordinator.dispatch(
            "export",
            {"project_id": project_id, "output_mode": "ALPHA_ONLY", "destination": str(alpha_path)},
        )
        self.assertEqual(alpha_result["bit_depth"], 16)
        with Image.open(alpha_path) as image:
            alpha_values = np.asarray(image)
        self.assertGreater(int(alpha_values.max()), 255)

    def test_protocol_returns_structured_errors_and_recovers_next_line(self) -> None:
        response = handle_request(self.coordinator, {"id": "bad", "method": "unknown"})
        self.assertFalse(response["ok"])
        self.assertIn("details", response["error"])

        input_stream = io.StringIO(
            "not json\n" + json.dumps({"id": "ok", "method": "health", "params": {}}) + "\n"
        )
        output_stream = io.StringIO()
        original = Coordinator
        try:
            # serve_stdio owns its coordinator; this still verifies line isolation.
            serve_stdio(input_stream, output_stream)
        finally:
            _ = original
        lines = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertFalse(lines[0]["ok"])
        self.assertTrue(lines[1]["ok"])


if __name__ == "__main__":
    unittest.main()

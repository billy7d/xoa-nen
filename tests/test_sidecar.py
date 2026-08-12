from __future__ import annotations

import io
import base64
import hashlib
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "sidecar"
if str(SIDECAR) not in sys.path:
    sys.path.insert(0, str(SIDECAR))

from cutout_sidecar.coordinator import Coordinator  # noqa: E402
from cutout_sidecar.exports import _decontaminate_edges, _resize_alpha_with_support, export_image  # noqa: E402
from cutout_sidecar.image_core import decode_canonical, load_canonical_png  # noqa: E402
from cutout_sidecar.legacy_v1 import artwork_alpha as legacy_artwork_alpha  # noqa: E402
from cutout_sidecar import model_runtime as model_runtime_module  # noqa: E402
from cutout_sidecar import models as model_registry  # noqa: E402
from cutout_sidecar.model_runtime import LocalModelRuntime  # noqa: E402
from cutout_sidecar.models import install_model_pack, list_model_manifests  # noqa: E402
from cutout_sidecar.processor import (  # noqa: E402
    _delta_e_2000,
    _lab_to_srgb,
    _srgb_to_lab,
    artwork_alpha,
    conservative_object_masks,
    magic_wand_selection,
)
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
    def test_ciede2000_matches_published_reference_pair(self) -> None:
        first = np.array([[[50.0, 2.6772, -79.7751]]], dtype=np.float32)
        second = np.array([50.0, 0.0, -82.7485], dtype=np.float32)
        self.assertAlmostEqual(float(_delta_e_2000(first, second)[0, 0]), 2.0425, places=4)

    def test_lab_conversion_round_trip_stays_within_one_rgb_level(self) -> None:
        rgb = np.array([[[12, 143, 231], [245, 238, 220]]], dtype=np.uint8)
        restored = np.rint(_lab_to_srgb(_srgb_to_lab(rgb))).astype(np.uint8)
        np.testing.assert_allclose(restored, rgb, atol=1)

    def test_background_is_removed_without_reanimating_source_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "art.png"
            rgb, source_alpha = artwork_fixture(source)
            result, diagnostics = artwork_alpha(rgb, source_alpha, tolerance=25, softness=10)
            self.assertLess(float(result[0, 0]), 0.01)
            self.assertGreater(float(result[22, 30]), 0.98)
            self.assertGreater(float(result[10, 113]), 0.98)
            self.assertLessEqual(float(np.max(result - source_alpha)), 1e-6)
            self.assertEqual(diagnostics["engine"], "hybrid-cutout-v3")
            self.assertEqual(diagnostics["source_alpha_contract"], "multiply")
            self.assertIn("selected_strategy", diagnostics)

    def test_multitone_background_and_low_contrast_subject_do_not_leak(self) -> None:
        height, width = 120, 160
        ramp = np.linspace(232, 248, width, dtype=np.float32)
        rgb = np.repeat(ramp[None, :, None], height, axis=0)
        rgb = np.repeat(rgb, 3, axis=2).astype(np.uint8)
        rgb[25:96, 35:126] = np.clip(
            rgb[25:96, 35:126].astype(np.int16) - 8, 0, 255
        ).astype(np.uint8)
        source_alpha = np.ones((height, width), dtype=np.float32)

        result, diagnostics = artwork_alpha(
            rgb, source_alpha, tolerance=30, softness=12, quality_preset="QUALITY"
        )

        self.assertLess(float(result[5, 80]), 0.02)
        self.assertGreater(float(result[60, 80]), 0.95)
        self.assertGreaterEqual(int(diagnostics["background_model"]["order"]), 1)

    def test_ambiguous_hole_is_preserved_when_outline_has_tiny_gap(self) -> None:
        rgb = np.full((96, 128, 3), 246, dtype=np.uint8)
        rgb[20:76, 28:34] = (35, 58, 62)
        rgb[20:76, 94:100] = (35, 58, 62)
        rgb[20:26, 28:100] = (35, 58, 62)
        rgb[70:76, 28:100] = (35, 58, 62)
        rgb[20, 63:65] = 246  # anti-alias-sized opening into a same-colour interior
        source_alpha = np.ones((96, 128), dtype=np.float32)

        result, _ = artwork_alpha(rgb, source_alpha, tolerance=25, softness=10)

        self.assertLess(float(result[5, 5]), 0.02)
        self.assertGreater(float(result[48, 64]), 0.95)

    def test_foreground_touching_one_border_does_not_poison_background_palette(self) -> None:
        rgb = np.full((100, 140, 3), 242, dtype=np.uint8)
        rgb[:74, 48:92] = (42, 57, 68)
        source_alpha = np.ones((100, 140), dtype=np.float32)

        result, diagnostics = artwork_alpha(rgb, source_alpha, tolerance=25, softness=10)

        self.assertLess(float(result[90, 8]), 0.02)
        self.assertGreater(float(result[8, 70]), 0.95)
        self.assertEqual(int(diagnostics["background_model"]["order"]), 0)

    def test_legacy_profile_is_pixel_exact_with_frozen_v1(self) -> None:
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, size=(73, 91, 3), dtype=np.uint8)
        source_alpha = rng.random((73, 91), dtype=np.float32)
        expected, _ = legacy_artwork_alpha(rgb, source_alpha, tolerance=31, softness=17)
        actual, diagnostics = artwork_alpha(
            rgb,
            source_alpha,
            tolerance=31,
            softness=17,
            engine_profile="LEGACY_V1",
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(diagnostics["selected_strategy"], "legacy_v1_exact")

    def test_hard_edge_noise_corner_and_source_alpha_gates(self) -> None:
        source_alpha = np.ones((96, 128), dtype=np.float32)
        rgb = np.full((96, 128, 3), 244, dtype=np.uint8)
        truth = np.zeros((96, 128), dtype=np.float32)
        rgb[20:78, 30:106] = (33, 68, 54)
        truth[20:78, 30:106] = 1.0
        alpha, _ = artwork_alpha(rgb, source_alpha, tolerance=30, softness=18)
        self.assertLessEqual(float(np.mean(np.abs(alpha - truth))), 0.005)

        rng = np.random.default_rng(7)
        noisy = np.clip(244 + rng.normal(0, 4, rgb.shape), 0, 255).astype(np.uint8)
        noisy[20:78, 30:106] = (33, 68, 54)
        alpha, _ = artwork_alpha(noisy, source_alpha, tolerance=30, softness=18)
        self.assertLessEqual(float(np.mean(np.abs(alpha - truth))), 0.01)

        corner = np.full((96, 128, 3), 244, dtype=np.uint8)
        corner[:48, :54] = (33, 68, 54)
        corner_truth = np.zeros((96, 128), dtype=np.float32)
        corner_truth[:48, :54] = 1.0
        alpha, _ = artwork_alpha(corner, source_alpha, tolerance=30, softness=18)
        intersection = np.count_nonzero((alpha > 0.5) & (corner_truth > 0.5))
        union = np.count_nonzero((alpha > 0.5) | (corner_truth > 0.5))
        self.assertGreaterEqual(intersection / max(1, union), 0.97)

        semi = source_alpha.copy()
        semi[20:78, 30:106] = 0.4
        alpha, _ = artwork_alpha(rgb, semi, tolerance=30, softness=18)
        self.assertLessEqual(float(np.max(alpha - semi)), 1e-7)
        self.assertAlmostEqual(float(np.mean(alpha[30:60, 40:90])), 0.4, places=5)

    def test_ai_conservative_candidate_restores_object_holes_and_preserves_handle_hole(self) -> None:
        height, width = 96, 128
        rgb = np.full((height, width, 3), 240, dtype=np.uint8)
        semantic = np.zeros((height, width), dtype=np.float32)
        semantic[20:82, 30:88] = 0.99
        semantic[8:25, 56:62] = 0.99  # Ống hút mảnh.
        semantic[35:76, 86:116] = 0.99
        semantic[43:68, 94:108] = 0.0  # Lỗ quai là background thật.
        source_alpha = np.ones((height, width), dtype=np.float32)
        source_alpha[26:78, 58:78] = 0.0  # Alpha đầu vào đã xoá nhầm thân.
        source_alpha[8:25, 56:62] = 0.0

        alpha, diagnostics, guidance = artwork_alpha(
            rgb,
            source_alpha,
            engine_profile="V3_AI_LOCAL",
            semantic_alpha=semantic,
            foreground_points=[{"x": 48.0, "y": 48.0}],
            return_guidance=True,
        )

        assert guidance is not None
        self.assertEqual(diagnostics["result_status"], "READY")
        self.assertGreater(float(alpha[12, 58]), 0.95)
        self.assertGreater(float(alpha[50, 68]), 0.95)
        self.assertLess(float(alpha[55, 101]), 0.05)
        self.assertLess(float(alpha[4, 4]), 0.01)
        self.assertFalse(np.any(alpha[guidance["sure_foreground"]] < 0.95))
        self.assertFalse(np.any(guidance["unknown"] & guidance["sure_foreground"]))

    def test_shift_prompt_adds_detached_detail_without_filling_negative_space(self) -> None:
        semantic = np.zeros((64, 96), dtype=np.float32)
        semantic[20:56, 24:62] = 0.99
        semantic[6:17, 70:75] = 0.99  # Chi tiết rời cần Shift-click.
        semantic[31:47, 48:57] = 0.0

        one_point = conservative_object_masks(semantic, foreground_points=[{"x": 36, "y": 36}])
        two_points = conservative_object_masks(
            semantic,
            foreground_points=[{"x": 36, "y": 36}, {"x": 72, "y": 10}],
        )

        self.assertFalse(bool(one_point["object_candidate"][10, 72]))
        self.assertTrue(bool(two_points["object_candidate"][10, 72]))
        self.assertFalse(bool(two_points["object_candidate"][38, 52]))
        self.assertTrue(bool(two_points["sure_background"][38, 52]))


class MagicWandTests(unittest.TestCase):
    def test_seed_patch_respects_clicked_side_of_a_boundary(self) -> None:
        rgb = np.full((48, 64, 3), 238, dtype=np.uint8)
        rgb[:, 16:] = (55, 70, 82)

        selection = magic_wand_selection(
            rgb, 15, 24, tolerance=12, softness=5, contiguous=True
        )

        self.assertGreater(float(selection[24, 15]), 0.98)
        self.assertLess(float(selection[24, 20]), 0.01)

    def test_contiguous_wand_stops_at_weak_edge_even_inside_tolerance(self) -> None:
        rgb = np.full((96, 128, 3), 230, dtype=np.uint8)
        rgb[20:76, 32:96] = 218

        contiguous = magic_wand_selection(rgb, 4, 4, tolerance=30, softness=12, contiguous=True)
        global_selection = magic_wand_selection(
            rgb, 4, 4, tolerance=30, softness=12, contiguous=False
        )

        self.assertGreater(float(contiguous[4, 4]), 0.98)
        self.assertLess(float(contiguous[48, 64]), 0.05)
        self.assertGreater(float(global_selection[48, 64]), 0.5)

    def test_wand_produces_fractional_antialiased_edge_coverage(self) -> None:
        rgb = np.full((72, 96, 3), 245, dtype=np.uint8)
        rgb[16:56, 28:68] = 70
        rgb[16:56, 27] = 200
        rgb[16:56, 26] = 228

        selection = magic_wand_selection(
            rgb, 4, 4, tolerance=30, softness=20, contiguous=True
        )

        self.assertGreater(float(selection[36, 4]), 0.98)
        self.assertLess(float(selection[36, 48]), 0.01)
        self.assertTrue(np.any((selection[34:39, 25:29] > 0.01) & (selection[34:39, 25:29] < 0.99)))


class PodEdgeTests(unittest.TestCase):
    def test_multimode_palette_decontaminates_each_edge_against_nearest_background(self) -> None:
        foreground = np.array([190.0, 50.0, 25.0], dtype=np.float32)
        backgrounds = np.array([[240.0, 240.0, 240.0], [24.0, 24.0, 24.0]], dtype=np.float32)
        observed = np.rint(0.5 * foreground + 0.5 * backgrounds).astype(np.uint8)[None, :, :]
        alpha = np.full((1, 2), 0.5, dtype=np.float32)

        palette_result = _decontaminate_edges(observed, alpha, backgrounds.tolist())
        median_result = _decontaminate_edges(observed, alpha, [132.0, 132.0, 132.0])

        palette_error = np.mean(np.abs(palette_result.astype(np.float32) - foreground))
        median_error = np.mean(np.abs(median_result.astype(np.float32) - foreground))
        self.assertLess(float(palette_error), float(median_error))

    def test_decontamination_never_changes_fully_opaque_or_transparent_pixels(self) -> None:
        rgb = np.array([[[10, 20, 30], [150, 80, 50], [90, 100, 110]]], dtype=np.uint8)
        alpha = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = _decontaminate_edges(rgb, alpha, [240.0, 240.0, 240.0])
        np.testing.assert_array_equal(result[:, [0, 2]], rgb[:, [0, 2]])

    def test_alpha_upscale_keeps_exact_output_size_and_transparent_hole(self) -> None:
        alpha = np.zeros((5, 7), dtype=np.float32)
        alpha[1:4, 1:6] = 1.0
        alpha[2, 3] = 0.0  # Lỗ quai/negative space không được sinh alpha.
        for scale in (2, 3, 4):
            enhanced = _resize_alpha_with_support(alpha, scale)
            self.assertEqual(enhanced.shape, (5 * scale, 7 * scale))
            self.assertTrue(np.all(np.isfinite(enhanced)))
            self.assertGreaterEqual(float(enhanced.min()), 0.0)
            self.assertLessEqual(float(enhanced.max()), 1.0)
            self.assertLess(float(enhanced[2 * scale, 3 * scale]), 0.01)

    def test_upscale_request_refuses_lanczos_when_no_qualified_model_exists(self) -> None:
        rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
        alpha = np.ones((8, 8), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "không dùng Lanczos giả AI"):
                export_image(
                    "POD_READY", Path(temporary) / "out.png", rgb, alpha, None,
                    settings={"upscale_mode": "FAITHFUL", "upscale_scale": 2},
                )

    def test_upscale_export_reports_exact_x2_x3_x4_dimensions_and_alpha(self) -> None:
        class FakeUpscaleRuntime:
            """Mock model để test contract export, không đại diện cho model AI phát hành."""

            def upscale_rgb(self, rgb, mode, scale, cancel_check=None):
                enhanced = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
                return enhanced, {
                    "status": "ok", "model_id": f"mock-{mode.lower()}-x{scale}",
                    "backend": "CPUExecutionProvider", "latency_ms": 1.0,
                }

        rgb = np.full((5, 7, 3), 128, dtype=np.uint8)
        alpha = np.ones((5, 7), dtype=np.float32)
        alpha[2, 3] = 0.0
        with tempfile.TemporaryDirectory() as temporary:
            for scale in (2, 3, 4):
                destination = Path(temporary) / f"x{scale}.png"
                result = export_image(
                    "POD_READY", destination, rgb, alpha, None,
                    settings={"upscale_mode": "FAITHFUL", "upscale_scale": scale},
                    runtime=FakeUpscaleRuntime(),
                )
                self.assertEqual((result["width"], result["height"]), (7 * scale, 5 * scale))
                self.assertEqual(result["model"], f"mock-faithful-x{scale}")
                with Image.open(destination) as image:
                    output_alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
                self.assertEqual(output_alpha.shape, (5 * scale, 7 * scale))
                self.assertEqual(int(output_alpha[2 * scale, 3 * scale]), 0)


class CoordinatorFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source = root / "nested unicode" / "áo POD.png"
        self.source.parent.mkdir()
        self.expected_rgb, self.source_alpha = artwork_fixture(self.source)
        self.coordinator = Coordinator(ProjectStore(root / "projects"))
        # Cô lập kho model của test để artifact local trên máy phát triển không đổi kết quả regression.
        self.coordinator.models_dir = root / "models"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_import_process_edit_undo_redo_preflight_and_exports(self) -> None:
        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        project_id = imported["project_id"]
        self.assertEqual(imported["schema_version"], "3.0.0")
        self.assertEqual(imported["canonical"]["decoded_pixel_hash"], imported["manifest"]["source"]["canonical_pixels_sha256"])
        self.assertTrue(Path(imported["preview_path"]).is_file())

        processed = self.coordinator.dispatch(
            "process_artwork",
            {"project_id": project_id, "tolerance": 25, "softness": 10, "quality_preset": "QUALITY"},
        )
        self.assertEqual(processed["process"]["content_mode"], "AUTO")
        self.assertEqual(processed["process"]["engine_profile"], "V3_BALANCED")
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

    def test_wand_preview_commit_and_subject_selection_contracts(self) -> None:
        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        project_id = imported["project_id"]
        processed = self.coordinator.dispatch(
            "process_artwork",
            {"project_id": project_id, "engine_profile": "V3_BALANCED"},
        )
        before = self.coordinator.store.read_alpha(project_id).copy()
        preview = self.coordinator.dispatch(
            "preview_magic_wand",
            {
                "project_id": project_id,
                "x": 2,
                "y": 2,
                "mode": "remove",
                "wand_algorithm": "SMART",
                "contiguous": True,
            },
        )
        self.assertGreater(preview["selected_pixel_count"], 0)
        self.assertTrue(Path(preview["preview_path"]).is_file())
        np.testing.assert_array_equal(self.coordinator.store.read_alpha(project_id), before)
        committed = self.coordinator.dispatch(
            "commit_magic_wand",
            {
                "project_id": project_id,
                "selection_id": preview["selection_id"],
                "mode": "remove",
                "wand_algorithm": "SMART",
            },
        )
        self.assertTrue(committed["history"]["can_undo"])
        self.assertFalse(Path(preview["preview_path"]).exists())

        subject_ids = [item["id"] for item in processed["process"]["subjects"]]
        selected = subject_ids[:1]
        result = self.coordinator.dispatch(
            "set_subject_selection",
            {"project_id": project_id, "selected_subject_ids": selected},
        )
        self.assertEqual(result["process"]["selected_subject_ids"], selected)
        self.assertEqual(result["process"]["subject_policy"], "SELECTED")

    def test_ai_profile_falls_back_explicitly_without_model_pack(self) -> None:
        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        result = self.coordinator.dispatch(
            "process_artwork",
            {"project_id": imported["project_id"], "engine_profile": "V3_AI_LOCAL"},
        )
        self.assertEqual(result["process"]["ai_models_used"], [])
        self.assertTrue(any("fallback" in warning.lower() for warning in result["warnings"]))
        self.assertIn("fallback", result["process"]["diagnostics"]["selected_strategy"])

    def test_cancel_enhanced_job_marks_job_and_sets_cancellation_event(self) -> None:
        cancel_event = threading.Event()
        job_id = "a" * 32
        # Mô phỏng job đang chạy để kiểm tra cancel không cần model/ảnh thật.
        self.coordinator._jobs[job_id] = {
            "job_id": job_id, "project_id": "test", "status": "RUNNING",
            "created_at": 0.0, "cancel_event": cancel_event, "result": None, "error": None,
        }
        result = self.coordinator.dispatch("cancel_job", {"job_id": job_id})
        self.assertTrue(cancel_event.is_set())
        self.assertEqual(result["status"], "CANCELLING")
        self.assertNotIn("cancel_event", result)

    def test_processing_manifest_persists_conservative_foreground_prompt(self) -> None:
        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        result = self.coordinator.dispatch(
            "process_artwork",
            {
                "project_id": imported["project_id"],
                "foreground_points": [{"x": 50.0, "y": 48.0}],
                "background_points": [],
                "protection_mode": "CONSERVATIVE",
                "shadow_policy": "REMOVE",
            },
        )
        process = result["process"]
        self.assertEqual(process["foreground_points"], [{"x": 50.0, "y": 48.0}])
        self.assertEqual(process["protection_mode"], "CONSERVATIVE")
        self.assertEqual(process["shadow_policy"], "REMOVE")
        self.assertGreater(float(self.coordinator.store.read_alpha(imported["project_id"])[48, 50]), 0.69)

    def test_v2_manifest_migration_preserves_alpha_history_and_does_not_reprocess(self) -> None:
        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        project_id = imported["project_id"]
        original_alpha = self.coordinator.store.read_alpha(project_id).copy()
        manifest_path = self.coordinator.store.path(project_id) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "2.2.0"
        manifest["history"] = [{"tool": "archived-v2-edit"}]
        manifest["history_cursor"] = 1
        manifest["processing"] = {
            "content_mode": "ARTWORK",
            "diagnostics": {"engine": "classical-artwork-v2"},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        migrated = self.coordinator.store.manifest(project_id)
        self.assertEqual(migrated["schema_version"], "3.0.0")
        self.assertEqual(migrated["history"], [{"tool": "archived-v2-edit"}])
        self.assertEqual(migrated["history_cursor"], 1)
        self.assertEqual(migrated["processing"]["engine_profile"], "V2_ARCHIVED_RESULT")
        np.testing.assert_array_equal(self.coordinator.store.read_alpha(project_id), original_alpha)


class ModelPackTests(unittest.TestCase):
    def _signed_pack(self, root: Path, *, corrupt_checksum: bool = False) -> Path:
        artifact = b"audited-onnx-placeholder"
        private_key = Ed25519PrivateKey.generate()
        public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        model_registry.TRUSTED_SIGNING_KEYS["test-key"] = base64.b64encode(public).decode()
        manifest = {
            "schema_version": "1.0.0",
            "model_id": "test-birefnet-lite",
            "revision": "fixed-test-revision",
            "role": "base_alpha_proposal",
            "adapter": "birefnet-v1",
            "commercial_pod_allowed": True,
            "redistribution_allowed": True,
            "runtime_remote_code_allowed": False,
            "qualified_backends": ["CPUExecutionProvider"],
            "signature_key_id": "test-key",
            "artifacts": [{
                "filename": "model.onnx",
                "sha256": "0" * 64 if corrupt_checksum else hashlib.sha256(artifact).hexdigest(),
                "size": len(artifact),
                "role": "base_alpha_proposal",
            }],
        }
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        manifest["signature_ed25519"] = base64.b64encode(private_key.sign(payload)).decode()
        pack = root / "test.cutout-modelpack"
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr("pack/manifest.json", json.dumps(manifest))
            archive.writestr("pack/model.onnx", artifact)
        return pack

    def tearDown(self) -> None:
        model_registry.TRUSTED_SIGNING_KEYS.pop("test-key", None)

    def test_signed_pack_installs_atomically_and_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models_dir = root / "models"
            installed = install_model_pack(self._signed_pack(root), models_dir)
            self.assertTrue(installed["installed"])
            self.assertEqual(installed["status"], "ready")
            self.assertTrue((models_dir / "test-birefnet-lite" / "model.onnx").is_file())
            ready = [item for item in list_model_manifests(models_dir) if item.get("installed")]
            self.assertEqual(len(ready), 1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "corrupt_or_missing_artifact"):
                install_model_pack(self._signed_pack(root, corrupt_checksum=True), root / "models")


class LocalModelRuntimeTests(unittest.TestCase):
    class FakeInput:
        def __init__(self, name: str, shape: list[int | None]) -> None:
            self.name = name
            self.shape = shape

    class FakeSession:
        def __init__(self, outputs: list[np.ndarray], inputs: list["LocalModelRuntimeTests.FakeInput"]) -> None:
            self.outputs = outputs
            self.inputs = inputs
            self.last_inputs: dict[str, np.ndarray] = {}

        def get_inputs(self) -> list["LocalModelRuntimeTests.FakeInput"]:
            return self.inputs

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def run(self, _output_names: list[str] | None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            self.last_inputs = inputs
            return self.outputs

    class FakeOrt:
        def __init__(self, session: "LocalModelRuntimeTests.FakeSession") -> None:
            self.session = session

        def get_available_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def InferenceSession(self, _path: str, providers: list[str]) -> "LocalModelRuntimeTests.FakeSession":
            self.session.providers = providers
            return self.session

    def setUp(self) -> None:
        model_runtime_module._SESSION_CACHE.clear()

    def tearDown(self) -> None:
        model_runtime_module._SESSION_CACHE.clear()

    @staticmethod
    def _manifest(
        root: Path,
        *,
        role: str,
        adapter: str,
        input_size: list[int],
        input_name: str = "image",
        prompt_input_name: str | None = None,
        point_input_name: str | None = None,
        point_label_input_name: str | None = None,
    ) -> dict[str, object]:
        manifest: dict[str, object] = {
            "model_id": f"test-{adapter}",
            "revision": "runtime-test-revision",
            "role": role,
            "adapter": adapter,
            "install_path": str(root),
            "artifacts": [{"filename": "model.onnx", "role": role}],
            "qualified_backends": ["CPUExecutionProvider"],
            "input_size": input_size,
            "input_layout": "NCHW",
            "normalization": "none",
            "input_name": input_name,
            "output_layout": "NCHW",
            "output_activation": "sigmoid",
            "output_semantics": "foreground",
        }
        if prompt_input_name:
            manifest["prompt_input_name"] = prompt_input_name
        if point_input_name:
            manifest["point_input_name"] = point_input_name
        if point_label_input_name:
            manifest["point_label_input_name"] = point_label_input_name
        return manifest

    def test_birefnet_proposal_decodes_logits_and_resizes_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logits = np.full((1, 1, 4, 4), -10.0, dtype=np.float32)
            logits[:, :, 1:3, 1:3] = 10.0
            session = self.FakeSession(
                [logits],
                [self.FakeInput("image", [1, 3, 4, 4])],
            )
            fake_ort = self.FakeOrt(session)
            manifest = self._manifest(
                root,
                role="base_alpha_proposal",
                adapter="birefnet-v1",
                input_size=[4, 4],
            )
            rgb = np.full((8, 10, 3), 128, dtype=np.uint8)
            runtime = LocalModelRuntime(root)
            with patch.object(runtime, "_ready", return_value=manifest), patch.object(
                model_runtime_module, "ort", fake_ort
            ):
                proposal, diagnostics = runtime.semantic_proposal(rgb)

            self.assertIsNotNone(proposal)
            assert proposal is not None
            self.assertEqual(proposal.shape, (8, 10))
            # Letterbox giữ tỷ lệ nên pixel biên nội suy mềm hơn bản resize méo cũ.
            self.assertGreater(float(proposal[3, 4]), 0.75)
            self.assertLess(float(proposal[0, 0]), 0.01)
            self.assertEqual(diagnostics["adapter"], "birefnet-v1")
            self.assertEqual(session.last_inputs["image"].shape, (1, 3, 4, 4))

    def test_lite_proposal_uses_overlapping_focus_tiles_after_foreground_click(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.FakeSession(
                [np.full((1, 1, 4, 4), 10.0, dtype=np.float32)],
                [self.FakeInput("image", [1, 3, 4, 4])],
            )
            fake_ort = self.FakeOrt(session)
            manifest = self._manifest(
                root,
                role="base_alpha_proposal",
                adapter="birefnet-v1",
                input_size=[4, 4],
            )
            rgb = np.full((8, 10, 3), 128, dtype=np.uint8)
            runtime = LocalModelRuntime(root)
            with patch.object(runtime, "_ready", return_value=manifest), patch.object(
                model_runtime_module, "ort", fake_ort
            ):
                proposal, diagnostics = runtime.semantic_proposal(
                    rgb,
                    foreground_points=[{"x": 5.0, "y": 4.0}],
                )

            self.assertIsNotNone(proposal)
            self.assertEqual(diagnostics["focus_tile_count"], 2)
            self.assertTrue(np.all(proposal > 0.99))

    def test_vitmatte_refines_unknown_band_and_preserves_known_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.FakeSession(
                [np.full((1, 1, 4, 4), 2.0, dtype=np.float32)],
                [self.FakeInput("image", [1, 4, 4, 4])],
            )
            fake_ort = self.FakeOrt(session)
            manifest = self._manifest(
                root,
                role="roi_matting",
                adapter="vitmatte-v1",
                input_size=[4, 4],
            )
            rgb = np.full((32, 32, 3), 120, dtype=np.uint8)
            alpha = np.zeros((32, 32), dtype=np.float32)
            alpha[8:24, 8:24] = 1.0
            alpha[7, 8:24] = 0.5
            source_alpha = np.ones((32, 32), dtype=np.float32)
            runtime = LocalModelRuntime(root)
            with patch.object(runtime, "_ready", return_value=manifest), patch.object(
                model_runtime_module, "ort", fake_ort
            ):
                refined, diagnostics = runtime.refine_unknown(rgb, alpha, source_alpha)

            self.assertEqual(refined.shape, alpha.shape)
            self.assertAlmostEqual(float(refined[16, 16]), 1.0, places=5)
            self.assertAlmostEqual(float(refined[0, 0]), 0.0, places=5)
            self.assertAlmostEqual(float(refined[7, 16]), 0.8808, places=3)
            self.assertEqual(diagnostics["adapter"], "vitmatte-v1")
            self.assertEqual(session.last_inputs["image"].shape, (1, 4, 4, 4))

    def test_sam2_topology_uses_mask_prompt_without_writing_fractional_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.FakeSession(
                [np.full((1, 1, 4, 4), 10.0, dtype=np.float32)],
                [
                    self.FakeInput("image", [1, 3, 4, 4]),
                    self.FakeInput("prompt", [1, 1, 4, 4]),
                ],
            )
            fake_ort = self.FakeOrt(session)
            manifest = self._manifest(
                root,
                role="conditional_topology",
                adapter="sam2-conditional-v1",
                input_size=[4, 4],
                prompt_input_name="prompt",
            )
            rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
            semantic = np.full((8, 8), 0.75, dtype=np.float32)
            runtime = LocalModelRuntime(root)
            with patch.object(runtime, "_ready", return_value=manifest), patch.object(
                model_runtime_module, "ort", fake_ort
            ):
                topology, diagnostics = runtime.topology_proposal(rgb, semantic)

            self.assertIsNotNone(topology)
            assert topology is not None
            self.assertTrue(np.all(topology > 0.99))
            self.assertEqual(diagnostics["membership_only"], True)
            self.assertEqual(session.last_inputs["prompt"].shape, (1, 1, 4, 4))

    def test_vitmatte_changes_only_explicit_unknown_band(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.FakeSession(
                [np.full((1, 1, 4, 4), 2.0, dtype=np.float32)],
                [self.FakeInput("image", [1, 4, 4, 4])],
            )
            fake_ort = self.FakeOrt(session)
            manifest = self._manifest(
                root,
                role="roi_matting",
                adapter="vitmatte-v1",
                input_size=[4, 4],
            )
            rgb = np.full((32, 32, 3), 120, dtype=np.uint8)
            source_alpha = np.ones((32, 32), dtype=np.float32)
            alpha = np.zeros((32, 32), dtype=np.float32)
            sure_foreground = np.zeros((32, 32), dtype=bool)
            sure_foreground[10:22, 10:22] = True
            unknown = np.zeros((32, 32), dtype=bool)
            unknown[8:10, 10:22] = True
            sure_background = ~(sure_foreground | unknown)
            alpha[sure_foreground] = 1.0
            runtime = LocalModelRuntime(root)
            with patch.object(runtime, "_ready", return_value=manifest), patch.object(
                model_runtime_module, "ort", fake_ort
            ):
                refined, diagnostics = runtime.refine_unknown(
                    rgb,
                    alpha,
                    source_alpha,
                    sure_foreground=sure_foreground,
                    sure_background=sure_background,
                    unknown=unknown,
                )

            self.assertTrue(diagnostics["external_trimap"])
            self.assertTrue(np.all(refined[sure_foreground] == 1.0))
            self.assertTrue(np.all(refined[sure_background] == 0.0))
            self.assertAlmostEqual(float(refined[9, 16]), 0.8808, places=3)

    def test_sam2_topology_forwards_foreground_and_background_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = self.FakeSession(
                [np.full((1, 1, 4, 4), 10.0, dtype=np.float32)],
                [
                    self.FakeInput("image", [1, 3, 4, 4]),
                    self.FakeInput("points", [1, None, 2]),
                    self.FakeInput("labels", [1, None]),
                ],
            )
            fake_ort = self.FakeOrt(session)
            manifest = self._manifest(
                root,
                role="conditional_topology",
                adapter="sam2-point-prompt-v1",
                input_size=[4, 4],
                point_input_name="points",
                point_label_input_name="labels",
            )
            rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
            semantic = np.full((8, 8), 0.75, dtype=np.float32)
            runtime = LocalModelRuntime(root)
            with patch.object(runtime, "_ready", return_value=manifest), patch.object(
                model_runtime_module, "ort", fake_ort
            ):
                topology, diagnostics = runtime.topology_proposal(
                    rgb,
                    semantic,
                    foreground_points=[{"x": 2.0, "y": 4.0}],
                    background_points=[{"x": 7.0, "y": 0.0}],
                )

            self.assertIsNotNone(topology)
            self.assertEqual(diagnostics["prompt_count"], 2)
            self.assertEqual(session.last_inputs["points"].shape, (1, 2, 2))
            np.testing.assert_array_equal(session.last_inputs["labels"], [[1.0, 0.0]])

if __name__ == "__main__":
    unittest.main()

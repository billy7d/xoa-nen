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
from PIL import Image, ImageCms, ImageDraw
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "sidecar"
if str(SIDECAR) not in sys.path:
    sys.path.insert(0, str(SIDECAR))

from cutout_sidecar.coordinator import Coordinator  # noqa: E402
from cutout_sidecar.exports import _decontaminate_edges, _resize_alpha_with_support, export_image, pod_clean_rgb  # noqa: E402
from cutout_sidecar.image_core import decode_canonical, load_canonical_png  # noqa: E402
from cutout_sidecar.legacy_v1 import artwork_alpha as legacy_artwork_alpha  # noqa: E402
from cutout_sidecar import model_runtime as model_runtime_module  # noqa: E402
from cutout_sidecar import models as model_registry  # noqa: E402
from cutout_sidecar import processor as processor_module  # noqa: E402
from cutout_sidecar import worker as worker_module  # noqa: E402
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
from cutout_sidecar.watermark_engine.mask import apply_stroke_to_mask, rasterize_stroke  # noqa: E402
from cutout_sidecar.watermark_engine.detector import detect_watermark  # noqa: E402
from cutout_sidecar.watermark_engine.inpaint import restore_roi_with_candidate  # noqa: E402
from cutout_sidecar.watermark_engine.pipeline import restore_watermark  # noqa: E402


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

    def test_explicit_prior_cutout_recovery_restores_object_and_preserves_handle_hole(self) -> None:
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
            subject_policy="SELECTED",
            source_alpha_mode="RECOVER_PRIOR_CUTOUT",
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
        all_detected = conservative_object_masks(
            semantic,
            foreground_points=[{"x": 36, "y": 36}],
            restrict_to_prompted_components=False,
        )

        self.assertFalse(bool(one_point["object_candidate"][10, 72]))
        self.assertTrue(bool(two_points["object_candidate"][10, 72]))
        self.assertTrue(bool(all_detected["object_candidate"][10, 72]))
        self.assertFalse(bool(two_points["object_candidate"][38, 52]))
        self.assertTrue(bool(two_points["sure_background"][38, 52]))

    def test_subject_selection_points_are_separate_from_protection_clicks(self) -> None:
        semantic = np.zeros((80, 120), dtype=np.float32)
        semantic[15:65, 10:45] = 0.99
        semantic[20:70, 75:110] = 0.99
        both = conservative_object_masks(
            semantic,
            foreground_points=[{"x": 25, "y": 40}],
            selection_points=[{"x": 25, "y": 40}, {"x": 90, "y": 45}],
        )
        one = conservative_object_masks(
            semantic,
            foreground_points=[{"x": 25, "y": 40}],
            selection_points=[{"x": 90, "y": 45}],
        )
        none = conservative_object_masks(
            semantic,
            selection_points=[],
        )

        self.assertEqual(both["selected_component_count"], 2)
        self.assertEqual(both["protected_component_count"], 1)
        self.assertTrue(bool(both["object_candidate"][40, 25]))
        self.assertTrue(bool(both["object_candidate"][45, 90]))
        self.assertFalse(bool(one["object_candidate"][40, 25]))
        self.assertTrue(bool(one["object_candidate"][45, 90]))
        self.assertFalse(bool(np.any(none["object_candidate"])))

    def test_transparent_thin_detail_stays_unknown_instead_of_hard_foreground(self) -> None:
        semantic = np.zeros((512, 512), dtype=np.float32)
        semantic[150:430, 120:390] = 0.99
        semantic[144:151, 145:365] = 0.99  # Rail trong suốt nối với thân vật thể.

        guidance = conservative_object_masks(semantic)

        self.assertEqual(guidance["trimap_radius"], 4)
        self.assertTrue(bool(guidance["sure_foreground"][250, 250]))
        self.assertFalse(bool(guidance["sure_foreground"][147, 250]))
        self.assertTrue(bool(guidance["unknown"][147, 250]))
        self.assertTrue(bool(guidance["sure_background"][20, 20]))

    def test_ai_profile_preserves_rgba_source_alpha_by_default(self) -> None:
        rgb = np.full((96, 96, 3), 180, dtype=np.uint8)
        rgb[20:76, 20:76] = (80, 40, 120)
        source_alpha = np.ones((96, 96), dtype=np.float32)
        source_alpha[30:66, 30:66] = 0.40
        semantic = np.zeros((96, 96), dtype=np.float32)
        semantic[20:76, 20:76] = 0.99

        result, diagnostics = artwork_alpha(
            rgb,
            source_alpha,
            engine_profile="V3_AI_LOCAL",
            semantic_alpha=semantic,
        )

        self.assertLessEqual(float(np.max(result - source_alpha)), 1e-6)
        self.assertEqual(diagnostics["source_alpha_mode"], "PRESERVE")
        self.assertEqual(diagnostics["source_alpha_contract"], "multiply")

    def test_prior_cutout_recovery_requires_explicit_mode(self) -> None:
        rgb = np.full((96, 96, 3), 180, dtype=np.uint8)
        rgb[20:76, 20:76] = (80, 40, 120)
        source_alpha = np.ones((96, 96), dtype=np.float32)
        source_alpha[30:66, 30:66] = 0.40
        semantic = np.zeros((96, 96), dtype=np.float32)
        semantic[20:76, 20:76] = 0.99

        result, diagnostics = artwork_alpha(
            rgb,
            source_alpha,
            engine_profile="V3_AI_LOCAL",
            semantic_alpha=semantic,
            source_alpha_mode="RECOVER_PRIOR_CUTOUT",
        )

        self.assertGreater(float(result[48, 48]), float(source_alpha[48, 48]))
        self.assertEqual(
            diagnostics["source_alpha_contract"],
            "conservative_restore_inside_candidate",
        )

    def test_all_detected_prompt_does_not_hide_other_component_review(self) -> None:
        rgb = np.full((128, 160, 3), 235, dtype=np.uint8)
        rgb[20:105, 15:65] = (70, 35, 120)
        rgb[25:110, 95:145] = (65, 40, 115)
        semantic = np.zeros((128, 160), dtype=np.float32)
        semantic[20:105, 15:65] = 0.99
        semantic[25:110, 95:145] = 0.99
        source_alpha = np.ones((128, 160), dtype=np.float32)

        def disagreeing_graphcut(*args: object, **_kwargs: object):
            legacy_proxy = np.asarray(args[1])
            return np.zeros(legacy_proxy.shape, dtype=bool), "ok"

        with patch.object(
            processor_module,
            "_graphcut_foreground",
            side_effect=disagreeing_graphcut,
        ):
            _, all_diagnostics = artwork_alpha(
                rgb,
                source_alpha,
                engine_profile="V3_AI_LOCAL",
                semantic_alpha=semantic,
                foreground_points=[{"x": 40.0, "y": 60.0}],
                subject_policy="ALL_DETECTED",
            )
            _, selected_diagnostics = artwork_alpha(
                rgb,
                source_alpha,
                engine_profile="V3_AI_LOCAL",
                semantic_alpha=semantic,
                foreground_points=[{"x": 40.0, "y": 60.0}],
                subject_policy="SELECTED",
            )
            _, partial_diagnostics = artwork_alpha(
                rgb,
                source_alpha,
                engine_profile="V3_AI_LOCAL",
                semantic_alpha=semantic,
                foreground_points=[{"x": 40.0, "y": 60.0}],
                selection_points=[{"x": 40.0, "y": 60.0}, {"x": 120.0, "y": 65.0}],
                subject_policy="SELECTED",
            )
            _, missed_diagnostics = artwork_alpha(
                rgb,
                source_alpha,
                engine_profile="V3_AI_LOCAL",
                semantic_alpha=semantic,
                foreground_points=[{"x": 159.0, "y": 0.0}],
                subject_policy="SELECTED",
            )

        self.assertTrue(all_diagnostics["needs_review"])
        self.assertFalse(all_diagnostics["selected_scope_protected"])
        self.assertFalse(selected_diagnostics["needs_review"])
        self.assertTrue(selected_diagnostics["selected_scope_protected"])
        self.assertTrue(partial_diagnostics["needs_review"])
        self.assertEqual(partial_diagnostics["selected_component_count"], 2)
        self.assertEqual(partial_diagnostics["protected_component_count"], 1)
        self.assertTrue(missed_diagnostics["needs_review"])
        self.assertFalse(missed_diagnostics["selected_scope_protected"])

        def graph_matches_left_only(*args: object, **_kwargs: object):
            legacy_proxy = np.asarray(args[1])
            graph = np.zeros(legacy_proxy.shape, dtype=bool)
            graph[:, : graph.shape[1] // 2] = True
            return graph, "ok"

        with patch.object(
            processor_module,
            "_graphcut_foreground",
            side_effect=graph_matches_left_only,
        ):
            _, selected_scope = artwork_alpha(
                rgb,
                source_alpha,
                engine_profile="V3_AI_LOCAL",
                semantic_alpha=semantic,
                selection_points=[{"x": 40.0, "y": 60.0}],
                subject_policy="SELECTED",
            )
            missing_semantic = semantic.copy()
            missing_semantic[25:110, 95:145] = 0.0
            _, incomplete_selection = artwork_alpha(
                rgb,
                source_alpha,
                engine_profile="V3_AI_LOCAL",
                semantic_alpha=missing_semantic,
                selection_points=[{"x": 40.0, "y": 60.0}, {"x": 120.0, "y": 65.0}],
                subject_policy="SELECTED",
            )
        self.assertFalse(selected_scope["needs_review"])
        self.assertEqual(selected_scope["disagreement_scope"], "selected")
        self.assertTrue(incomplete_selection["selection_mapping_failed"])
        self.assertTrue(incomplete_selection["needs_protection"])


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
    def test_pod_preview_with_icc_profile_returns_rgb(self) -> None:
        rgb = np.full((4, 5, 3), 128, dtype=np.uint8)
        alpha = np.ones((4, 5), dtype=np.float32)
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

        preview_rgb = pod_clean_rgb(rgb, alpha, icc, None)

        self.assertEqual(preview_rgb.shape, rgb.shape)
        self.assertEqual(preview_rgb.dtype, np.uint8)

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

    def _mock_watermark_ai(self):
        original_request = self.coordinator.worker.request

        def request(method: str, params: dict[str, object]) -> dict[str, object]:
            if method != "restore_watermark":
                return original_request(method, params)
            # Mô phỏng output AI để test phiên/undo không phụ thuộc artifact ONNX bên ngoài.
            image_path = Path(str(params["image_path"]))
            mask_path = Path(str(params["mask_path"]))
            output_path = Path(str(params["output_path"]))
            with Image.open(image_path) as image:
                rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
            mask = np.load(mask_path, allow_pickle=False).astype(np.float32)
            output = rgb.copy()
            changed = mask > 0.01
            output[changed] = np.array([38, 168, 123], dtype=np.uint8)
            Image.fromarray(output, "RGB").save(output_path)
            ys, xs = np.where(changed)
            bounds = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            return {
                "output_path": str(output_path),
                "bounds": bounds,
                "diagnostics": {
                    "algorithm_version": "watermark-restore-v3-ai-local",
                    "selected": {"route": "AI_FAST", "model_id": "test-local-ai"},
                    "bounds": bounds,
                },
                "worker_pid": 12345,
            }

        return patch.object(self.coordinator.worker, "request", side_effect=request)

    def test_import_process_edit_undo_redo_preflight_and_exports(self) -> None:
        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        project_id = imported["project_id"]
        self.assertEqual(imported["schema_version"], "3.1.0")
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

    def test_manual_watermark_retouch_keeps_native_size_and_undo_redo(self) -> None:
        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        project_id = imported["project_id"]
        before = self.coordinator.store.read_working_rgb(project_id).copy()
        with self._mock_watermark_ai():
            result = self.coordinator.dispatch(
                "remove_watermark",
                {
                    "project_id": project_id,
                    "mode": "MANUAL",
                    "points": [{"x": 61, "y": 47}, {"x": 64, "y": 47}],
                    "radius": 5,
                },
            )
        after = self.coordinator.store.read_working_rgb(project_id)
        self.assertEqual(after.shape, before.shape)
        self.assertTrue(result["retouch"]["watermark_removed"])
        self.assertFalse(np.array_equal(before, after))

        undone = self.coordinator.dispatch("undo", {"project_id": project_id})
        np.testing.assert_array_equal(self.coordinator.store.read_working_rgb(project_id), before)
        self.assertTrue(undone["history"]["can_redo"])
        self.assertFalse(undone["retouch"]["watermark_removed"])
        self.coordinator.dispatch("redo", {"project_id": project_id})
        np.testing.assert_array_equal(self.coordinator.store.read_working_rgb(project_id), after)

        destination = Path(self.temporary.name) / "exports" / "retouched.png"
        exported = self.coordinator.dispatch(
            "export",
            {"project_id": project_id, "output_mode": "MASTER_SOURCE_FAITHFUL", "destination": str(destination)},
        )
        self.assertEqual((exported["width"], exported["height"]), (128, 96))
        with Image.open(destination) as image:
            self.assertEqual(image.size, (128, 96))

    def test_watermark_session_mask_hardness_subtract_and_tile_history(self) -> None:
        soft_low, _ = rasterize_stroke(
            (80, 80),
            [{"x": 12 + index, "y": 30} for index in range(40)],
            radius=9,
            hardness=0.1,
            feather=0,
        )
        soft_hard, _ = rasterize_stroke(
            (80, 80),
            [{"x": 12 + index, "y": 30} for index in range(40)],
            radius=9,
            hardness=1.0,
            feather=0,
        )
        self.assertGreater(float(np.max(np.abs(soft_low - soft_hard))), 0.1)

        base = np.zeros((80, 80), dtype=np.float32)
        added, _, _ = apply_stroke_to_mask(
            base,
            [{"x": 12 + index, "y": 42} for index in range(40)],
            radius=10,
            hardness=0.8,
            feather=2,
            mode="ADD",
        )
        subtracted, _, _ = apply_stroke_to_mask(
            added,
            [{"x": 28 + index, "y": 42} for index in range(12)],
            radius=8,
            hardness=1.0,
            feather=1,
            mode="SUBTRACT",
        )
        self.assertLess(int(np.count_nonzero(subtracted > 0.01)), int(np.count_nonzero(added > 0.01)))

        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        project_id = imported["project_id"]
        session = self.coordinator.dispatch(
            "begin_watermark_session",
            {
                "project_id": project_id,
                "quality": "BALANCED",
                "feather": 4,
                "expand": "LOW",
            },
        )
        stroke = [
            {"x": 35 + index * 0.045, "y": 42 + (index % 9) * 0.08}
            for index in range(1000)
        ]
        updated = self.coordinator.dispatch(
            "update_watermark_mask",
            {
                "project_id": project_id,
                "session_id": session["session_id"],
                "mode": "ADD",
                "points": stroke,
                "radius": 6,
                "hardness": 0.35,
                "feather": 4,
            },
        )
        self.assertGreater(updated["mask_pixel_count"], 0)
        self.assertTrue(Path(updated["mask_preview_path"]).is_file())
        before = self.coordinator.store.read_working_rgb(project_id).copy()
        with self._mock_watermark_ai():
            preview = self.coordinator.dispatch(
                "preview_watermark",
                {
                    "project_id": project_id,
                    "session_id": session["session_id"],
                    "quality": "FAST",
                },
            )
            self.assertEqual(preview["status"], "READY")
            self.assertTrue(Path(preview["preview_path"]).is_file())
            np.testing.assert_array_equal(self.coordinator.store.read_working_rgb(project_id), before)
            with Image.open(preview["preview_path"]) as image:
                preview_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            committed = self.coordinator.dispatch(
                "commit_watermark",
                {
                    "project_id": project_id,
                    "session_id": session["session_id"],
                    "quality": "FAST",
                },
            )
        after = self.coordinator.store.read_working_rgb(project_id).copy()
        self.assertEqual(after.shape, before.shape)
        np.testing.assert_array_equal(after, preview_rgb)
        self.assertTrue(committed["retouch"]["watermark_removed"])
        self.assertFalse(Path(updated["mask_preview_path"]).exists())
        manifest = self.coordinator.store.manifest(project_id)
        entry = manifest["history"][manifest["history_cursor"] - 1]
        self.assertTrue(entry.get("rgb_tiles"))
        self.assertNotIn("retouch_snapshots", entry)
        x0, y0, x1, y1 = entry["bounds"]
        outside = np.ones(before.shape[:2], dtype=bool)
        outside[y0:y1, x0:x1] = False
        np.testing.assert_array_equal(after[outside], before[outside])

        undone = self.coordinator.dispatch("undo", {"project_id": project_id})
        self.assertFalse(undone["retouch"]["watermark_removed"])
        np.testing.assert_array_equal(self.coordinator.store.read_working_rgb(project_id), before)

    def test_watermark_preview_is_invalidated_by_mask_or_project_changes(self) -> None:
        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        project_id = imported["project_id"]
        session = self.coordinator.dispatch(
            "begin_watermark_session", {"project_id": project_id, "quality": "FAST"}
        )
        updated = self.coordinator.dispatch(
            "update_watermark_mask",
            {
                "project_id": project_id,
                "session_id": session["session_id"],
                "mode": "ADD",
                "points": [{"x": 61, "y": 47}],
                "radius": 5,
                "hardness": 1.0,
                "feather": 0,
            },
        )
        with self._mock_watermark_ai():
            preview = self.coordinator.dispatch(
                "preview_watermark",
                {"project_id": project_id, "session_id": session["session_id"], "quality": "FAST"},
            )
        self.assertEqual(preview["status"], "READY")

        changed_mask = self.coordinator.dispatch(
            "update_watermark_mask",
            {
                "project_id": project_id,
                "session_id": session["session_id"],
                "mode": "SUBTRACT",
                "points": [{"x": 61, "y": 47}],
                "radius": 2,
                "hardness": 1.0,
                "feather": 0,
            },
        )
        self.assertEqual(changed_mask["status"], "EDITING")
        self.assertIsNone(changed_mask["preview_path"])

        with self._mock_watermark_ai():
            self.coordinator.dispatch(
                "preview_watermark",
                {"project_id": project_id, "session_id": session["session_id"], "quality": "FAST"},
            )
        self.coordinator.dispatch(
            "apply_brush",
            {
                "project_id": project_id,
                "points": [{"x": 4, "y": 4}],
                "radius": 2,
                "hardness": 1.0,
                "opacity": 1.0,
                "mode": "remove",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "Ảnh đã thay đổi"):
            self.coordinator.dispatch(
                "commit_watermark",
                {"project_id": project_id, "session_id": session["session_id"]},
            )

    def test_watermark_restoration_replaces_core_and_only_feathers_edges(self) -> None:
        original = np.zeros((7, 7, 3), dtype=np.uint8)
        candidate = np.full((7, 7, 3), 200, dtype=np.uint8)
        mask = np.zeros((7, 7), dtype=np.float32)
        mask[3, 3] = 0.35
        mask[3, 2] = 0.18
        restored = restore_roi_with_candidate(original, mask, candidate, (0, 0, 7, 7))
        np.testing.assert_array_equal(restored[3, 3], [200, 200, 200])
        self.assertTrue(np.all((restored[3, 2] > 0) & (restored[3, 2] < 200)))
        np.testing.assert_array_equal(restored[0, 0], [0, 0, 0])

    def test_auto_detector_finds_sparkle_without_selecting_the_scene(self) -> None:
        height, width = 256, 320
        yy, xx = np.mgrid[:height, :width]
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(54 + xx * 0.22 + yy * 0.08, 0, 255)
        rgb[:, :, 1] = np.clip(76 + xx * 0.12 + yy * 0.10, 0, 255)
        rgb[:, :, 2] = np.clip(48 + xx * 0.08 + yy * 0.06, 0, 255)
        image = Image.fromarray(rgb, "RGB")
        draw = ImageDraw.Draw(image)
        size, x0, y0 = 28, 276, 216
        angles = np.linspace(0.0, np.pi * 2.0, 240, endpoint=False)
        # Fixture dùng exponent khác detector để test không chỉ lặp lại đúng template nội bộ.
        points = [
            (
                x0 + size / 2 + (size - 7) / 2 * np.sign(np.cos(angle)) * abs(np.cos(angle)) ** 2.8,
                y0 + size / 2 + (size - 7) / 2 * np.sign(np.sin(angle)) * abs(np.sin(angle)) ** 2.8,
            )
            for angle in angles
        ]
        draw.polygon(points, fill=(192, 193, 176))

        detection = detect_watermark(np.asarray(image, dtype=np.uint8))

        self.assertEqual(detection.diagnostics["detector"], "GEMINI_SPARKLE_NCC")
        x1, y1, x2, y2 = detection.diagnostics["bounds"]
        self.assertLessEqual(abs(x1 - x0), 3)
        self.assertLessEqual(abs(y1 - y0), 3)
        self.assertLess(float(np.mean(detection.confidence > 0.01)), 0.02)

    def test_auto_detector_rejects_scene_wide_classical_mask(self) -> None:
        blocks = ((np.indices((192, 224)).sum(axis=0) // 4) % 2 * 255).astype(np.uint8)
        rgb = np.dstack((blocks, blocks, blocks))

        detection = detect_watermark(rgb)

        self.assertEqual(int(np.count_nonzero(detection.confidence)), 0)
        self.assertTrue(detection.diagnostics["safety_rejected"])

    def test_watermark_restore_requires_local_ai_and_never_routes_to_algorithm(self) -> None:
        class FakeLocalAiRuntime:
            def __init__(self) -> None:
                self.roles: list[str] = []

            def has_role(self, role: str) -> bool:
                return role in {"watermark_inpaint_fast", "watermark_inpaint_quality"}

            def inpaint_rgb(self, rgb: np.ndarray, _mask: np.ndarray, role: str, **_kwargs: object):
                self.roles.append(role)
                if role == "watermark_inpaint_quality":
                    return None, {"status": "inference_failed", "role": role}
                return np.full_like(rgb, 76), {"status": "ok", "model_id": "test-local-ai"}

        rgb = np.full((24, 24, 3), 230, dtype=np.uint8)
        mask = np.zeros((24, 24), dtype=np.float32)
        mask[8:16, 8:16] = 1.0
        runtime = FakeLocalAiRuntime()
        with patch(
            "cutout_sidecar.watermark_engine.pipeline.score_candidate",
            return_value={"overall": 0.95, "changed_mean": 12.0},
        ):
            restored, bounds, diagnostics = restore_watermark(
                rgb, mask, quality="MAXIMUM", runtime=runtime
            )
        self.assertEqual(runtime.roles, ["watermark_inpaint_quality", "watermark_inpaint_fast"])
        self.assertEqual(diagnostics["algorithm_version"], "watermark-restore-v3.1-ai-local-gated")
        self.assertEqual(diagnostics["selected"]["route"], "AI_FAST")
        self.assertTrue(all(str(item["route"]).startswith("AI_") for item in diagnostics["attempts"]))
        self.assertEqual(bounds, (8, 8, 16, 16))
        np.testing.assert_array_equal(restored[0, 0], rgb[0, 0])
        np.testing.assert_array_equal(restored[12, 12], [76, 76, 76])

        with self.assertRaisesRegex(RuntimeError, "Cần model AI local lấp nền watermark"):
            restore_watermark(rgb, mask, quality="BALANCED", runtime=None)

        with patch(
            "cutout_sidecar.watermark_engine.pipeline.score_candidate",
            return_value={"overall": 0.41, "changed_mean": 12.0},
        ), self.assertRaisesRegex(RuntimeError, "chưa tái tạo nền đạt quality gate"):
            restore_watermark(rgb, mask, quality="BALANCED", runtime=runtime)

    def test_watermark_preview_without_ai_model_keeps_project_unchanged(self) -> None:
        imported = self.coordinator.dispatch("import_image", {"path": str(self.source)})
        project_id = imported["project_id"]
        before = self.coordinator.store.read_working_rgb(project_id).copy()
        session = self.coordinator.dispatch(
            "begin_watermark_session", {"project_id": project_id, "quality": "FAST"}
        )
        self.coordinator.dispatch(
            "update_watermark_mask",
            {
                "project_id": project_id,
                "session_id": session["session_id"],
                "mode": "ADD",
                "points": [{"x": 61, "y": 47}],
                "radius": 5,
                "hardness": 1.0,
                "feather": 0,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "Cần model AI local lấp nền watermark"):
            self.coordinator.dispatch(
                "preview_watermark",
                {"project_id": project_id, "session_id": session["session_id"], "quality": "FAST"},
            )
        np.testing.assert_array_equal(self.coordinator.store.read_working_rgb(project_id), before)

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
        self.assertEqual(len(result["process"]["subject_selection_points"]), len(selected))

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
        self.assertEqual(migrated["schema_version"], "3.1.0")
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
            self.assertEqual(installed["status"], "runtime_ready_quality_pending")
            self.assertTrue(installed["runtime_ready"])
            self.assertFalse(installed["quality_qualified"])
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
        def __init__(
            self,
            outputs: list[np.ndarray],
            inputs: list["LocalModelRuntimeTests.FakeInput"],
            output_sequence: list[list[np.ndarray]] | None = None,
        ) -> None:
            self.outputs = outputs
            self.inputs = inputs
            self.output_sequence = output_sequence
            self.last_inputs: dict[str, np.ndarray] = {}
            self.run_count = 0

        def get_inputs(self) -> list["LocalModelRuntimeTests.FakeInput"]:
            return self.inputs

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def run(self, _output_names: list[str] | None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            output_index = self.run_count
            self.run_count += 1
            self.last_inputs = inputs
            if self.output_sequence:
                return self.output_sequence[min(output_index, len(self.output_sequence) - 1)]
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

    def test_lama_inpaint_accepts_byte_range_output(self) -> None:
        """LaMa ONNX trả RGB 0..255 nên không được clip thành trắng."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = np.zeros((1, 3, 4, 4), dtype=np.float32)
            output[:, 0, :, :] = 64.0
            output[:, 1, :, :] = 128.0
            output[:, 2, :, :] = 255.0
            session = self.FakeSession(
                [output],
                [
                    self.FakeInput("image", [1, 3, 4, 4]),
                    self.FakeInput("mask", [1, 1, 4, 4]),
                ],
            )
            manifest = self._manifest(
                root,
                role="watermark_inpaint_fast",
                adapter="lama-v1",
                input_size=[4, 4],
            )
            manifest.update(
                {
                    "image_input_name": "image",
                    "mask_input_name": "mask",
                    "mask_input_layout": "NCHW",
                    "output_range": "byte_0_255",
                }
            )
            runtime = LocalModelRuntime(root)
            rgb = np.full((4, 4, 3), 180, dtype=np.uint8)
            mask = np.zeros((4, 4), dtype=np.float32)
            mask[1:3, 1:3] = 0.35
            with patch.object(runtime, "_ready", return_value=manifest), patch.object(
                model_runtime_module, "ort", self.FakeOrt(session)
            ):
                restored, diagnostics = runtime.inpaint_rgb(rgb, mask)

            self.assertIsNotNone(restored)
            assert restored is not None
            np.testing.assert_array_equal(restored[0, 0], np.array([64, 128, 255], dtype=np.uint8))
            self.assertEqual(diagnostics["status"], "ok")
            self.assertEqual(session.last_inputs["mask"].shape, (1, 1, 4, 4))
            self.assertEqual(set(np.unique(session.last_inputs["mask"])), {0.0, 1.0})

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

    def test_prompt_focus_tile_recovers_detail_missing_from_full_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = np.full((1, 1, 4, 4), -10.0, dtype=np.float32)
            full[0, 0, 2:, :2] = 10.0
            focused = np.full((1, 1, 4, 4), -10.0, dtype=np.float32)
            focused[0, 0, 1:4, 1:4] = 10.0
            session = self.FakeSession(
                [full],
                [self.FakeInput("image", [1, 3, 4, 4])],
                output_sequence=[[full], [focused]],
            )
            fake_ort = self.FakeOrt(session)
            manifest = self._manifest(
                root,
                role="base_alpha_proposal",
                adapter="birefnet-v1",
                input_size=[4, 4],
            )
            rgb = np.full((12, 12, 3), 128, dtype=np.uint8)
            runtime = LocalModelRuntime(root)
            with patch.object(runtime, "_ready", return_value=manifest), patch.object(
                model_runtime_module, "ort", fake_ort
            ):
                proposal, diagnostics = runtime.semantic_proposal(
                    rgb,
                    foreground_points=[{"x": 9.0, "y": 2.0}],
                )

            assert proposal is not None
            self.assertGreater(float(proposal[2, 9]), 0.50)
            self.assertEqual(diagnostics["prompt_focus_tile_count"], 1)
            self.assertEqual(session.run_count, 2)

    def test_auto_merge_keeps_parent_identity_and_enclosed_negative_space(self) -> None:
        base = np.zeros((96, 96), dtype=np.float32)
        base[10:80, 8:48] = 0.99
        base[50:86, 70:88] = 0.99
        base[61:75, 75:83] = 0.0  # Lỗ âm kín thuộc component nhỏ.
        detail = base.copy()
        detail[10:80, 48:55] = 0.99  # Tăng sai từ component lớn.
        detail[44:86, 64:94] = 0.99  # Rail đúng nối component nhỏ.
        detail[61:75, 75:83] = 0.99  # Model tile cố lấp lỗ âm.

        merged = model_runtime_module._merge_supported_detail(
            base,
            detail,
            parent_point=(72.0, 55.0),
        )

        self.assertLess(float(np.max(merged[20:70, 48:55])), 0.01)
        self.assertGreater(float(merged[55, 68]), 0.10)
        self.assertLess(float(np.max(merged[61:75, 75:83])), 0.01)

        prompted = model_runtime_module._merge_prompted_detail(
            base,
            detail,
            prompt_point=(72.0, 55.0),
        )
        self.assertGreater(float(prompted[55, 68]), 0.10)
        self.assertLess(float(np.max(prompted[61:75, 75:83])), 0.01)

        two_holes = np.zeros((96, 96), dtype=np.float32)
        two_holes[12:84, 12:84] = 0.99
        two_holes[30:44, 24:38] = 0.0
        two_holes[54:68, 58:72] = 0.0
        filled = np.where(two_holes > 0.02, two_holes, 0.99).astype(np.float32)
        restored_one = model_runtime_module._merge_prompted_detail(
            two_holes,
            filled,
            prompt_point=(30.0, 36.0),
        )
        self.assertGreater(float(restored_one[36, 30]), 0.50)
        self.assertLess(float(np.max(restored_one[54:68, 58:72])), 0.01)

        edge_weight = model_runtime_module._detail_tile_weight(
            9,
            9,
            (False, False, True, False),
        )
        self.assertGreater(float(edge_weight[4, 8]), 0.99)
        self.assertLess(float(edge_weight[4, 0]), 0.01)

    def test_quality_proposal_automatically_runs_component_detail_tile(self) -> None:
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
                proposal, diagnostics = runtime.semantic_proposal(rgb, auto_detail=True)

            self.assertIsNotNone(proposal)
            self.assertTrue(diagnostics["auto_detail"])
            self.assertGreaterEqual(diagnostics["auto_focus_tile_count"], 1)
            self.assertEqual(session.run_count, diagnostics["focus_tile_count"] + 1)

    def test_worker_marks_quality_pipeline_degraded_when_matte_fails(self) -> None:
        class FakeRuntime:
            def semantic_proposal(self, rgb: np.ndarray, *_args: object, **_kwargs: object):
                semantic = np.zeros(rgb.shape[:2], dtype=np.float32)
                semantic[8:24, 8:24] = 0.99
                return semantic, {"status": "ok", "model_id": "fake-base"}

            def topology_proposal(self, *_args: object, **_kwargs: object):
                return None, {"status": "model_not_installed"}

            def refine_unknown(self, _rgb: np.ndarray, alpha: np.ndarray, *_args: object, **_kwargs: object):
                return alpha, {"status": "inference_failed", "error": "synthetic"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            output = root / "alpha.npy"
            Image.new("RGB", (32, 32), (220, 220, 220)).save(source)
            with patch.object(worker_module, "LocalModelRuntime", return_value=FakeRuntime()):
                result = worker_module.process_request(
                    {
                        "method": "process_artwork",
                        "params": {
                            "canonical_path": str(source),
                            "output_path": str(output),
                            "models_dir": str(root / "models"),
                            "engine_profile": "V3_AI_LOCAL",
                            "quality_preset": "QUALITY",
                        },
                    }
                )

        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["ai_pipeline_status"], "degraded")
        self.assertEqual(diagnostics["result_status"], "NEEDS_PROTECTION")
        self.assertTrue(diagnostics["needs_review"])

    def test_worker_accepts_no_unknown_roi_and_reports_model_quality(self) -> None:
        class FakeRuntime:
            matte_quality = True
            topology_pending = False

            def semantic_proposal(self, rgb: np.ndarray, *_args: object, **_kwargs: object):
                semantic = np.zeros(rgb.shape[:2], dtype=np.float32)
                semantic[8:24, 8:24] = 0.99
                return semantic, {
                    "status": "ok",
                    "model_id": "fake-base",
                    "quality_qualified": True,
                }

            def topology_proposal(self, _rgb: np.ndarray, semantic: np.ndarray, *_args: object, **_kwargs: object):
                if self.topology_pending:
                    return np.ones_like(semantic), {
                        "status": "ok",
                        "model_id": "fake-topology",
                        "quality_qualified": False,
                    }
                return None, {"status": "model_not_installed"}

            def refine_unknown(self, _rgb: np.ndarray, alpha: np.ndarray, *_args: object, **_kwargs: object):
                return alpha, {
                    "status": "no_unknown_roi",
                    "model_id": "fake-matte",
                    "quality_qualified": self.matte_quality,
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            output = root / "alpha.npy"
            Image.new("RGB", (32, 32), (220, 220, 220)).save(source)
            with patch.object(worker_module, "LocalModelRuntime", return_value=FakeRuntime()):
                result = worker_module.process_request(
                    {
                        "method": "process_artwork",
                        "params": {
                            "canonical_path": str(source),
                            "output_path": str(output),
                            "models_dir": str(root / "models"),
                            "engine_profile": "V3_AI_LOCAL",
                            "quality_preset": "QUALITY",
                        },
                    }
                )
            FakeRuntime.matte_quality = False
            output_pending = root / "alpha-pending.npy"
            with patch.object(worker_module, "LocalModelRuntime", return_value=FakeRuntime()):
                pending_result = worker_module.process_request(
                    {
                        "method": "process_artwork",
                        "params": {
                            "canonical_path": str(source),
                            "output_path": str(output_pending),
                            "models_dir": str(root / "models"),
                            "engine_profile": "V3_AI_LOCAL",
                            "quality_preset": "QUALITY",
                        },
                    }
                )
            FakeRuntime.matte_quality = True
            FakeRuntime.topology_pending = True
            output_topology = root / "alpha-topology.npy"
            with patch.object(worker_module, "LocalModelRuntime", return_value=FakeRuntime()):
                topology_result = worker_module.process_request(
                    {
                        "method": "process_artwork",
                        "params": {
                            "canonical_path": str(source),
                            "output_path": str(output_topology),
                            "models_dir": str(root / "models"),
                            "engine_profile": "V3_AI_LOCAL",
                            "quality_preset": "QUALITY",
                        },
                    }
                )

        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["ai_pipeline_status"], "complete")
        self.assertEqual(diagnostics["ai_quality_status"], "qualified")
        self.assertNotEqual(diagnostics.get("fallback_reason"), "matting_no_unknown_roi")
        self.assertEqual(pending_result["diagnostics"]["ai_quality_status"], "pending")
        self.assertEqual(topology_result["diagnostics"]["ai_quality_status"], "pending")
        self.assertTrue(topology_result["diagnostics"]["ai_runtime"]["topology"]["applied"])

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

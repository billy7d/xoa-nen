from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageOps


SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_GUARANTEED_PIXELS = 40_000_000
MAX_GUARANTEED_EDGE = 10_000
PREVIEW_MAX_EDGE = 2048


@dataclass(slots=True)
class CanonicalImage:
    source_path: Path
    source_file_sha256: str
    canonical_pixels_sha256: str
    width: int
    height: int
    source_format: str
    source_color_mode: str
    original_orientation: int
    icc_profile: bytes | None
    source_alpha: np.ndarray
    rgb: np.ndarray
    conversion_flags: list[str]

    def to_manifest(self, image_id: str) -> dict[str, Any]:
        return {
            "image_id": image_id,
            "source_file_sha256": self.source_file_sha256,
            "canonical_pixels_sha256": self.canonical_pixels_sha256,
            "width": self.width,
            "height": self.height,
            "source_format": self.source_format,
            "source_color_mode": self.source_color_mode,
            "original_orientation": self.original_orientation,
            "canonical_orientation": 1,
            "icc_hash": hashlib.sha256(self.icc_profile).hexdigest()
            if self.icc_profile
            else None,
            "has_source_alpha": bool(np.any(self.source_alpha < 1.0)),
            "conversion_flags": self.conversion_flags,
            "decoder_version": "pillow-canonical-v1",
            "guaranteed_size": self.width * self.height <= MAX_GUARANTEED_PIXELS
            and max(self.width, self.height) <= MAX_GUARANTEED_EDGE,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _orientation(image: Image.Image) -> int:
    try:
        return int(image.getexif().get(274, 1))
    except (AttributeError, TypeError, ValueError):
        return 1


def _convert_cmyk_to_srgb(image: Image.Image, icc_profile: bytes | None) -> Image.Image:
    if icc_profile:
        try:
            source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
            target_profile = ImageCms.createProfile("sRGB")
            return ImageCms.profileToProfile(
                image,
                source_profile,
                target_profile,
                outputMode="RGB",
            )
        except (ImageCms.PyCMSError, OSError, ValueError):
            pass
    return image.convert("RGB")


def decode_canonical(path: str | Path) -> CanonicalImage:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {source_path}")

    source_hash = _sha256_file(source_path)
    with Image.open(source_path) as opened:
        source_format = (opened.format or "").upper()
        if source_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Ứng dụng chỉ hỗ trợ PNG, JPEG và static WebP; nhận được {source_format or opened.mode}"
            )
        if getattr(opened, "is_animated", False):
            raise ValueError("Ứng dụng không hỗ trợ ảnh động/multi-frame")

        original_orientation = _orientation(opened)
        source_mode = opened.mode
        icc_profile = opened.info.get("icc_profile")
        conversion_flags: list[str] = []
        oriented = ImageOps.exif_transpose(opened)

        if oriented.mode == "CMYK":
            oriented = _convert_cmyk_to_srgb(oriented, icc_profile)
            icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            conversion_flags.append("SOURCE_CONVERTED_CMYK_TO_SRGB")

        if oriented.mode in {"RGBA", "LA"}:
            rgba = oriented.convert("RGBA")
            rgba_array = np.asarray(rgba, dtype=np.uint8)
            rgb = np.ascontiguousarray(rgba_array[:, :, :3])
            source_alpha = np.ascontiguousarray(rgba_array[:, :, 3], dtype=np.float32) / 255.0
        elif oriented.mode == "P" and "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            rgba_array = np.asarray(rgba, dtype=np.uint8)
            rgb = np.ascontiguousarray(rgba_array[:, :, :3])
            source_alpha = np.ascontiguousarray(rgba_array[:, :, 3], dtype=np.float32) / 255.0
            conversion_flags.append("PALETTE_EXPANDED")
        else:
            if oriented.mode not in {"RGB", "L", "P"}:
                conversion_flags.append(f"SOURCE_CONVERTED_{oriented.mode}_TO_RGB")
            rgb = np.ascontiguousarray(np.asarray(oriented.convert("RGB"), dtype=np.uint8))
            source_alpha = np.ones(rgb.shape[:2], dtype=np.float32)

    height, width = rgb.shape[:2]
    pixel_digest = hashlib.sha256()
    pixel_digest.update(width.to_bytes(8, "little"))
    pixel_digest.update(height.to_bytes(8, "little"))
    pixel_digest.update(rgb.tobytes(order="C"))
    pixel_digest.update(source_alpha.astype("<f4", copy=False).tobytes(order="C"))

    if width * height > MAX_GUARANTEED_PIXELS or max(width, height) > MAX_GUARANTEED_EDGE:
        conversion_flags.append("SIZE_BEST_EFFORT")

    return CanonicalImage(
        source_path=source_path,
        source_file_sha256=source_hash,
        canonical_pixels_sha256=pixel_digest.hexdigest(),
        width=width,
        height=height,
        source_format=source_format,
        source_color_mode=source_mode,
        original_orientation=original_orientation,
        icc_profile=icc_profile,
        source_alpha=source_alpha,
        rgb=rgb,
        conversion_flags=conversion_flags,
    )


def save_canonical_png(canonical: CanonicalImage, destination: Path) -> None:
    alpha = np.rint(np.clip(canonical.source_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba = np.dstack((canonical.rgb, alpha))
    image = Image.fromarray(rgba, "RGBA")
    options: dict[str, Any] = {"compress_level": 6}
    if canonical.icc_profile:
        options["icc_profile"] = canonical.icc_profile
    image.save(destination, format="PNG", **options)


def load_canonical_png(path: Path) -> tuple[np.ndarray, np.ndarray, bytes | None]:
    with Image.open(path) as image:
        icc = image.info.get("icc_profile")
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    source_alpha = np.ascontiguousarray(rgba[:, :, 3], dtype=np.float32) / 255.0
    return rgb, source_alpha, icc


def inference_srgb_copy(rgb: np.ndarray, icc_profile: bytes | None) -> tuple[np.ndarray, bool]:
    """Return an sRGB inference buffer without mutating canonical source pixels."""
    if not icc_profile:
        return np.ascontiguousarray(rgb), False
    try:
        source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
        target_profile = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(
            Image.fromarray(rgb, "RGB"),
            source_profile,
            target_profile,
            outputMode="RGB",
        )
        return np.ascontiguousarray(np.asarray(converted, dtype=np.uint8)), True
    except (ImageCms.PyCMSError, OSError, ValueError):
        # A broken profile must not make the editor unusable. Diagnostics expose
        # the fallback so the output is never silently presented as color-managed.
        return np.ascontiguousarray(rgb), False


def preview_size(width: int, height: int, max_edge: int = PREVIEW_MAX_EDGE) -> tuple[int, int]:
    scale = min(1.0, max_edge / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def save_preview(
    rgb: np.ndarray,
    alpha: np.ndarray,
    destination: Path,
    max_edge: int = PREVIEW_MAX_EDGE,
) -> tuple[int, int]:
    height, width = alpha.shape
    target = preview_size(width, height, max_edge)
    rgb_image = Image.fromarray(rgb, "RGB")
    alpha_image = Image.fromarray(
        np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8), "L"
    )
    if target != (width, height):
        rgb_image = rgb_image.resize(target, Image.Resampling.LANCZOS)
        alpha_image = alpha_image.resize(target, Image.Resampling.LANCZOS)
    rgba = rgb_image.convert("RGBA")
    rgba.putalpha(alpha_image)
    rgba.save(destination, format="PNG", compress_level=3)
    return target

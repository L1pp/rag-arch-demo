"""Small page-region contract shared by renderers, model tools and provenance.

Regions use the displayed page, including its existing rotation and CropBox.
Coordinates are fractions of that page, measured from its top-left corner.
Identity uses explicit coordinates; no file digests or image fingerprints.
"""
from __future__ import annotations

import math


REGION_COORDINATE_SYSTEM = "display_page_normalized_top_left"
REGION_TARGET_EDGE = 1600
REGION_MAX_RENDER_SCALE = 8.0


def normalize_region(region):
    """Validate a four-number region, preserving its exact floating value."""
    if region is None:
        return None
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise ValueError("区域必须是四个归一化坐标 [左, 上, 右, 下]。")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in region):
        raise ValueError("区域坐标必须是有限实数，不能使用布尔值。")
    try:
        values = tuple(0.0 if value == 0 else float(value) for value in region)
    except (ValueError, OverflowError) as exc:
        raise ValueError("区域坐标必须是有限实数。") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("区域坐标必须是有限实数。")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("区域坐标须在 0 到 1 内，且左小于右、上小于下。")
    return values


def page_image_id(source_ref, page, region=None):
    """Keep whole-page IDs compatible and distinguish every normalized crop."""
    values = normalize_region(region)
    identifier = f"PAGE:{source_ref}:{page}"
    return identifier if values is None else identifier + ":REGION:" + ",".join(repr(v) for v in values)


def validate_page_result_identity(result, source_ref, page, region=None):
    """Reject substituted pages/crops before attaching a requested identity.

    Whole-page legacy results may omit ``region``. A crop must explicitly
    return the same normalized coordinates; its caller must never invent them
    from the request when an older renderer silently ignored that request.
    """
    expected_region = normalize_region(region)
    if not isinstance(result, dict):
        raise ValueError("页图返回内容无效。")
    if (not isinstance(source_ref, str) or not source_ref
            or result.get("source_ref") != source_ref):
        raise ValueError("页图返回来源与请求不一致。")
    if (type(page) is not int or page < 1 or type(result.get("page")) is not int
            or result["page"] != page):
        raise ValueError("页图返回页码与请求不一致。")
    try:
        actual_region = normalize_region(result.get("region"))
    except ValueError:
        raise ValueError("页图返回区域与请求不一致。") from None
    if actual_region != expected_region:
        raise ValueError("页图返回区域与请求不一致。")
    return result


def render_page_image(original, scale=1.5, region=None, *, max_pixels=4_000_000,
                      max_edge=2400, max_bytes=8 * 1024 * 1024):
    """Render the displayed region directly; never upscale a stored preview."""
    import pymupdf

    values = normalize_region(region)
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not 0.25 <= scale <= 4 or not math.isfinite(scale)):
        raise ValueError("页面缩放必须在 0.25 到 4 之间。")
    page_rect = original.rect
    page_width, page_height = page_rect.width, page_rect.height
    if not (page_width > 0 and page_height > 0):
        raise ValueError("页面尺寸无效，无法读取。")
    if values is None:
        clip = pymupdf.Rect(page_rect)
        requested = float(scale)
    else:
        x0, y0, x1, y1 = values
        clip = pymupdf.Rect(page_rect.x0 + x0 * page_width, page_rect.y0 + y0 * page_height,
                            page_rect.x0 + x1 * page_width, page_rect.y0 + y1 * page_height)
        if clip.width < 0.01 or clip.height < 0.01:
            raise ValueError("区域过小，请选择更大的页面区域。")
        requested = max(float(scale), min(REGION_MAX_RENDER_SCALE,
                                          REGION_TARGET_EDGE / max(clip.width, clip.height)))
    # get_pixmap clips in displayed coordinates and honours the page rotation.
    # Text extraction uses unrotated coordinates, so transform only its clip.
    text_clip = clip * original.derotation_matrix if values is not None else None
    render_scale = min(requested, (max_edge - 2) / max(clip.width, clip.height),
                       math.sqrt(max_pixels / (clip.width * clip.height)) * 0.99)
    for _ in range(4):
        pixmap = original.get_pixmap(matrix=pymupdf.Matrix(render_scale, render_scale),
                                     clip=clip, colorspace=pymupdf.csRGB, alpha=False)
        if (pixmap.width < 1 or pixmap.height < 1
                or max(pixmap.width, pixmap.height) > max_edge
                or pixmap.width * pixmap.height > max_pixels):
            render_scale *= 0.7
            continue
        png = pixmap.tobytes("png")
        if len(png) <= max_bytes:
            break
        render_scale *= 0.7
    else:
        raise ValueError("页面图像过大或区域过小，无法在图像限制内读取。")
    return {
        "text": original.get_text(clip=text_clip).strip(),
        "text_scope": "region" if values is not None else "page",
        "image_bytes": png, "mime_type": "image/png", "width": pixmap.width,
        "height": pixmap.height, "rendered_scale": render_scale,
        "region": list(values) if values is not None else None,
        "page_width": page_width, "page_height": page_height,
        "rotation": original.rotation, "coordinate_system": REGION_COORDINATE_SYSTEM,
    }

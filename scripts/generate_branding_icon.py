"""Build the official square "N" brand icon from the master banner.

The canonical "N" glyph is the one in the official wordmark banner
(``app/static/logo.png``) — a stylized sharp serif-italic "N" with a
notched upper-left serif. It is NOT a stock font glyph, so this script
never renders text with PIL/FreeType and never loads an external TTF.

Pipeline (fully deterministic):
1. Load ``app/static/logo.png`` (white-on-transparent wordmark).
2. Split the leading "N" from the "i" with a minimal-ink vertical seam
   (dynamic programming over x in [150, 280), y in [0, 360)).
3. Crop the "N" to its tight ink bounding box -> glyph master.
4. Center the master on a 512x512 pure-black squircle (rx=120), scaled so
   the glyph height is ~58% of the canvas.

Outputs (all under ``app/static/``):
- ``icon.png`` — 512x512 PNG for the Stremio manifest/catalog,
- ``favicon.png`` — 256x256 PNG,
- ``apple-touch-icon.png`` — 180x180 PNG,
- ``favicon.ico`` — multi-resolution ICO (16/32/48),
- ``favicon.svg`` — black squircle with the exact glyph embedded as a
  PNG data URI (pixel-identical, zero font-fallback risk).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageDraw

CANVAS = 512
CORNER_RADIUS = 120
GLYPH_HEIGHT_RATIO = 0.58

# Seam search window (banner pixels) isolating "N" from the following "i".
_SEAM_X0, _SEAM_X1 = 150, 280
_SEAM_Y0, _SEAM_Y1 = 0, 360
_SEAM_ANCHOR_X = 215


def _isolate_n_glyph(banner: Image.Image) -> Image.Image:
    """Crop the leading "N" out of the wordmark banner via a minimal-ink seam."""
    banner = banner.convert("RGBA")
    alpha = banner.split()[3]
    width, _height = banner.size
    ncol = _SEAM_X1 - _SEAM_X0
    nrow = _SEAM_Y1 - _SEAM_Y0

    ink = [
        [1 if alpha.getpixel((x, y)) > 10 else 0 for x in range(_SEAM_X0, _SEAM_X1)]
        for y in range(_SEAM_Y0, _SEAM_Y1)
    ]
    predecessor = [[0] * ncol for _ in range(nrow)]
    # First row: strong anchor keeps the seam in the N/i gap.
    best = [c * 1000 + abs((_SEAM_X0 + i) - _SEAM_ANCHOR_X) for i, c in enumerate(ink[0])]
    for y in range(1, nrow):
        current = [0.0] * ncol
        for x in range(ncol):
            parent = x
            parent_cost = best[x]
            if x > 0 and best[x - 1] < parent_cost:
                parent, parent_cost = x - 1, best[x - 1]
            if x + 1 < ncol and best[x + 1] < parent_cost:
                parent, parent_cost = x + 1, best[x + 1]
            predecessor[y][x] = parent
            current[x] = (
                parent_cost + ink[y][x] * 1000 + abs((_SEAM_X0 + x) - _SEAM_ANCHOR_X) * 0.02
            )
        best = current

    x = min(range(ncol), key=lambda i: best[i])
    seam = [0] * nrow
    for y in range(nrow - 1, -1, -1):
        seam[y] = _SEAM_X0 + x
        x = predecessor[y][x]

    mask = Image.new("L", (width, banner.size[1]), 0)
    mask_pixels = mask.load()
    for y in range(_SEAM_Y0, _SEAM_Y1):
        for x in range(seam[y - _SEAM_Y0]):
            mask_pixels[x, y] = 255
    foreground = Image.new("RGBA", banner.size, (0, 0, 0, 0))
    foreground.paste(banner, mask=mask)

    ink_alpha = foreground.split()[3]
    xs = [
        x
        for x in range(width)
        for y in range(_SEAM_Y0, _SEAM_Y1)
        if ink_alpha.getpixel((x, y)) > 10
    ]
    ys = [
        y
        for y in range(_SEAM_Y0, _SEAM_Y1)
        for x in range(width)
        if ink_alpha.getpixel((x, y)) > 10
    ]
    return foreground.crop((min(xs), min(ys), max(xs) + 1, max(ys) + 1))


def _compose_icon(glyph: Image.Image, size: int) -> Image.Image:
    """Center the glyph on a black squircle of the given pixel size."""
    target_h = round(size * GLYPH_HEIGHT_RATIO)
    scale = target_h / glyph.size[1]
    glyph_scaled = glyph.resize(
        (round(glyph.size[0] * scale), target_h), Image.LANCZOS
    )
    canvas_size = (size, size)
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    layer.alpha_composite(
        glyph_scaled,
        ((size - glyph_scaled.size[0]) // 2, (size - glyph_scaled.size[1]) // 2),
    )
    opaque = Image.new("RGB", (size, size), (0, 0, 0))
    opaque.paste(layer.convert("RGB"), mask=layer.split()[3])
    radius = round(CORNER_RADIUS * size / CANVAS)
    corner_mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(corner_mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=radius, fill=255
    )
    final = Image.new("RGB", (size, size), (0, 0, 0))
    final.paste(opaque, mask=corner_mask)
    return final


def generate_branding_icons() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    static_dir = root_dir / "app" / "static"
    logo_path = static_dir / "logo.png"
    if not logo_path.is_file():
        raise FileNotFoundError(f"Master banner not found: {logo_path}")
    print(f"Isolating official 'N' glyph from master banner: {logo_path}")
    glyph = _isolate_n_glyph(Image.open(str(logo_path)))
    print(f"Glyph master isolated: {glyph.size[0]}x{glyph.size[1]}")

    icon = _compose_icon(glyph, 512)
    icon_path = static_dir / "icon.png"
    icon.save(str(icon_path), "PNG")
    print(f"Square icon generated: {icon_path} ({icon_path.stat().st_size} bytes)")

    favicon_png_path = static_dir / "favicon.png"
    _compose_icon(glyph, 256).save(str(favicon_png_path), "PNG")
    print(f"Raster PNG favicon generated: {favicon_png_path}")

    apple_path = static_dir / "apple-touch-icon.png"
    _compose_icon(glyph, 180).save(str(apple_path), "PNG")
    print(f"Apple touch icon generated: {apple_path}")

    ico_path = static_dir / "favicon.ico"
    icon.save(str(ico_path), format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    print(f"Multi-resolution ICO generated: {ico_path}")

    buffer = io.BytesIO()
    icon.save(buffer, format="PNG")
    embedded = base64.b64encode(buffer.getvalue()).decode("ascii")
    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS}" height="{CANVAS}"'
        f' viewBox="0 0 {CANVAS} {CANVAS}">\n'
        f'<rect width="{CANVAS}" height="{CANVAS}" rx="{CORNER_RADIUS}" fill="#000000"/>\n'
        f'<image href="data:image/png;base64,{embedded}"/>\n'
        f"</svg>\n"
    )
    svg_path = static_dir / "favicon.svg"
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"SVG favicon generated: {svg_path} ({len(svg_content)} bytes)")


if __name__ == "__main__":
    generate_branding_icons()

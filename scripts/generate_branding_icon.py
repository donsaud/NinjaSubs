"""Generate the official square "N" brand icon from Serif BlackItalic.

Uses ``app/static/fonts/SerifBlackItalic.ttf`` (the official brand variant of
the Serif Black family) and renders the capital "N" glyph as a crisp white
mark centered on a pure-black squircle (512x512 canvas, rx=120), with the
glyph scaled to ~58% of the canvas height for balanced safe margins.

Outputs (all under ``app/static/``):
- ``favicon.svg`` — exact vector path of the "N" glyph (raw font-unit path
  inside a ``<g transform>``; no baked rounding, no system-font fallback),
- ``icon.png`` — 512x512 PNG for the Stremio manifest/catalog,
- ``favicon.png`` — 256x256 PNG,
- ``apple-touch-icon.png`` — 180x180 PNG,
- ``favicon.ico`` — multi-resolution ICO (16/32/48).
"""

from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

CANVAS = 512
CORNER_RADIUS = 120
GLYPH_HEIGHT_RATIO = 0.58
GLYPH_CHAR = "N"


def _load_brand_font(fonts_dir: Path) -> Path:
    font_path = fonts_dir / "SerifBlackItalic.ttf"
    if not font_path.is_file():
        raise FileNotFoundError(f"Serif BlackItalic font not found in {fonts_dir}")
    font = TTFont(str(font_path))
    family = font["name"].getDebugName(1)
    if family != "Serif BlackItalic":
        raise ValueError(f"Unexpected font family {family!r} in {font_path}")
    if ord(GLYPH_CHAR) not in font.getBestCmap():
        raise ValueError(f"Glyph {GLYPH_CHAR!r} missing from {font_path}")
    print(f"Loading official brand font: {font_path} ({family})")
    return font_path


def _glyph_geometry(font_path: Path) -> tuple[str, float, float, float]:
    """Return (raw SVG path, glyph-center-x, glyph-center-y, scale)."""
    tt_font = TTFont(str(font_path))
    glyph_set = tt_font.getGlyphSet()
    glyph_name = tt_font.getBestCmap()[ord(GLYPH_CHAR)]
    glyph = glyph_set[glyph_name]

    bounds_pen = BoundsPen(glyph_set)
    glyph.draw(bounds_pen)
    minx, miny, maxx, maxy = bounds_pen.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    scale = (CANVAS * GLYPH_HEIGHT_RATIO) / (maxy - miny)

    svg_pen = SVGPathPen(glyph_set)
    glyph.draw(svg_pen)
    return svg_pen.getCommands(), cx, cy, scale


def _write_svg(static_dir: Path, path_data: str, cx: float, cy: float, scale: float) -> None:
    tx = CANVAS / 2 - cx * scale
    ty = CANVAS / 2 + cy * scale
    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS}" height="{CANVAS}"'
        f' viewBox="0 0 {CANVAS} {CANVAS}">\n'
        f'<rect width="{CANVAS}" height="{CANVAS}" rx="{CORNER_RADIUS}" fill="#000000"/>\n'
        f'<g transform="translate({tx:.4f} {ty:.4f}) scale({scale:.6f} -{scale:.6f})">\n'
        f'  <path d="{path_data}" fill="#ffffff"/>\n'
        f"</g>\n"
        f"</svg>\n"
    )
    svg_path = static_dir / "favicon.svg"
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"Vector SVG favicon generated: {svg_path} ({len(svg_content)} bytes)")


def _render_raster(size: int, font_path: Path) -> Image.Image:
    """Render the white "N" on a black squircle at the given pixel size."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = round(CORNER_RADIUS * size / CANVAS)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=(0, 0, 0, 255))

    target_h = size * GLYPH_HEIGHT_RATIO
    lo, hi, best = 10, size * 3, int(size * 0.7)
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = ImageFont.truetype(str(font_path), mid)
        bbox = draw.textbbox((0, 0), GLYPH_CHAR, font=trial)
        if bbox[3] - bbox[1] < target_h:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    font = ImageFont.truetype(str(font_path), best)
    bbox = draw.textbbox((0, 0), GLYPH_CHAR, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
        GLYPH_CHAR,
        font=font,
        fill=(255, 255, 255, 255),
    )

    background = Image.new("RGB", (size, size), (0, 0, 0))
    background.paste(img.convert("RGB"), mask=img.split()[3])
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=radius, fill=255
    )
    final = Image.new("RGB", (size, size), (0, 0, 0))
    final.paste(background, mask=mask)
    return final


def generate_branding_icons() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    static_dir = root_dir / "app" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    font_path = _load_brand_font(static_dir / "fonts")
    path_data, cx, cy, scale = _glyph_geometry(font_path)
    _write_svg(static_dir, path_data, cx, cy, scale)

    icon = _render_raster(512, font_path)
    icon_path = static_dir / "icon.png"
    icon.save(str(icon_path), "PNG")
    print(f"Raster PNG icon generated: {icon_path} ({icon_path.stat().st_size} bytes)")

    favicon_png_path = static_dir / "favicon.png"
    _render_raster(256, font_path).save(str(favicon_png_path), "PNG")
    print(f"Raster PNG favicon generated: {favicon_png_path}")

    apple_path = static_dir / "apple-touch-icon.png"
    _render_raster(180, font_path).save(str(apple_path), "PNG")
    print(f"Apple touch icon generated: {apple_path}")

    ico_path = static_dir / "favicon.ico"
    icon.save(str(ico_path), format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    print(f"Multi-resolution ICO generated: {ico_path}")


if __name__ == "__main__":
    generate_branding_icons()

"""Generate vector SVG favicon and 256x256 PNG icon from Hoshiko Satsuki font."""

from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


def generate_branding_icons():
    root_dir = Path(__file__).resolve().parent.parent
    static_dir = root_dir / "app" / "static"
    fonts_dir = static_dir / "fonts"
    static_dir.mkdir(parents=True, exist_ok=True)

    font_path = fonts_dir / "Hoshiko-Satsuki.ttf"
    if not font_path.is_file():
        font_path = fonts_dir / "HoshikoSatsuki.ttf"
    if not font_path.is_file():
        raise FileNotFoundError(f"Hoshiko Satsuki font not found in {fonts_dir}")

    print(f"Loading font from: {font_path}")

    # 1. Generate SVG Favicon using fontTools SVGPathPen
    tt_font = TTFont(str(font_path))
    glyph_set = tt_font.getGlyphSet()
    cmap = tt_font.getBestCmap()
    glyph_name = cmap.get(ord("N"), "N")
    glyph = glyph_set[glyph_name]

    bounds_pen = BoundsPen(glyph_set)
    glyph.draw(bounds_pen)
    minx, miny, maxx, maxy = bounds_pen.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    glyph_h = maxy - miny

    svg_pen = SVGPathPen(glyph_set)
    glyph.draw(svg_pen)
    path_data = svg_pen.getCommands()

    view_size = 256
    target_glyph_h = 160.0
    scale = target_glyph_h / glyph_h

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_size} {view_size}" width="{view_size}" height="{view_size}">
  <rect x="2" y="2" width="{view_size - 4}" height="{view_size - 4}" rx="56" ry="56" fill="#141619" stroke="#22262b" stroke-width="2"/>
  <g transform="translate({view_size / 2:.1f}, {view_size / 2:.1f}) scale({scale:.4f}, -{scale:.4f}) translate({-cx:.2f}, {-cy:.2f})">
    <path d="{path_data}" fill="#ffffff"/>
  </g>
</svg>
"""

    svg_path = static_dir / "favicon.svg"
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"Vector SVG favicon generated: {svg_path} ({len(svg_content)} bytes)")

    # 2. Generate 256x256 High-Resolution PNG using Pillow
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([2, 2, 254, 254], radius=56, fill="#141619", outline="#22262b", width=2)

    font_size = 175
    pil_font = ImageFont.truetype(str(font_path), font_size)
    bbox = draw.textbbox((0, 0), "N", font=pil_font)
    bx0, by0, bx1, by1 = bbox
    x = (256 - (bx0 + bx1)) / 2.0
    y = (256 - (by0 + by1)) / 2.0
    draw.text((x, y), "N", font=pil_font, fill="#ffffff")

    png_path = static_dir / "icon.png"
    img.save(str(png_path), "PNG")
    print(f"Raster PNG icon generated: {png_path} ({png_path.stat().st_size} bytes)")

    favicon_png_path = static_dir / "favicon.png"
    img.save(str(favicon_png_path), "PNG")
    print(
        f"Raster PNG favicon generated: {favicon_png_path} ({favicon_png_path.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    generate_branding_icons()

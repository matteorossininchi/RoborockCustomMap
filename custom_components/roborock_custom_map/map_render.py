"""Background-override rendering for Roborock maps.

Everything here is synchronous and CPU/IO bound; run it via
hass.async_add_executor_job.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import io
import logging
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageOps
from vacuum_map_parser_base.config.color import Color, ColorsPalette, SupportedColor
from vacuum_map_parser_base.config.drawable import Drawable
from vacuum_map_parser_base.config.image_config import ImageConfig
from vacuum_map_parser_base.config.size import Size, Sizes
from vacuum_map_parser_base.map_data import ImageDimensions, Point
from vacuum_map_parser_roborock.map_data_parser import RoborockMapDataParser

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import STORAGE_DIR

from .const import (
    CARPETS_APP,
    CONF_CARPETS,
    CONF_CLEANED_AREA,
    CONF_PATH_COLOR,
    CONF_ROOM_COLORS,
    DOMAIN,
    MAX_BG_DIMENSION,
    MAX_COMPOSITE_DIMENSION,
    MAX_COMPOSITE_PIXELS,
    MAX_SCALED_DIMENSION,
    MAX_SCALED_PIXELS,
    MAX_UPLOAD_BYTES,
)

_LOGGER = logging.getLogger(__name__)

_TRANSPARENT: Color = (0, 0, 0, 0)
_OPAQUE_WHITE: Color = (255, 255, 255, 255)

_BASE_COLORS = (
    SupportedColor.MAP_OUTSIDE,
    SupportedColor.MAP_INSIDE,
    SupportedColor.MAP_WALL,
    SupportedColor.MAP_WALL_V2,
    SupportedColor.GREY_WALL,
    SupportedColor.SCAN,
    SupportedColor.CARPETS,
    SupportedColor.UNKNOWN,
    SupportedColor.NEW_DISCOVERED_AREA,
)

_WALL_COLORS = (
    SupportedColor.MAP_WALL,
    SupportedColor.MAP_WALL_V2,
    SupportedColor.GREY_WALL,
)

ALLOWED_FORMATS = {"png": "png", "jpeg": "jpg", "webp": "webp"}


class InvalidBackgroundImage(Exception):
    """Uploaded background failed validation; error_key maps to a flow error."""

    def __init__(self, error_key: str) -> None:
        super().__init__(error_key)
        self.error_key = error_key


def _transparent_room_colors() -> dict[str, Color]:
    return dict.fromkeys(ColorsPalette.ROOM_COLORS, _TRANSPARENT)


def _overlay_palette(path_color: Color | list[int] | None = None) -> ColorsPalette:
    """Palette rendering only drawables: all base pixels transparent."""
    colors: dict[SupportedColor, Color] = dict.fromkeys(_BASE_COLORS, _TRANSPARENT)
    if path_color is not None:
        colors[SupportedColor.PATH] = _rgb(path_color)
    return ColorsPalette(colors, _transparent_room_colors())


def _mask_palette() -> ColorsPalette:
    """Palette rendering only physical walls (and the charger) opaque."""
    colors: dict[SupportedColor, Color] = dict.fromkeys(_BASE_COLORS, _TRANSPARENT)
    for wall_color in _WALL_COLORS:
        colors[wall_color] = _OPAQUE_WHITE
    colors[SupportedColor.CHARGER] = _OPAQUE_WHITE
    colors[SupportedColor.CHARGER_OUTLINE] = _OPAQUE_WHITE
    return ColorsPalette(colors, _transparent_room_colors())


def _create_parser(
    palette: ColorsPalette,
    drawables: list[Drawable],
    map_scale: int,
    mop_width: float | None = None,
) -> RoborockMapDataParser:
    sizes = {
        k: v * map_scale for k, v in Sizes.SIZES.items() if k != Size.MOP_PATH_WIDTH
    }
    if mop_width is not None:
        sizes[Size.MOP_PATH_WIDTH] = mop_width
    return RoborockMapDataParser(
        palette,
        Sizes(sizes),
        drawables,
        ImageConfig(scale=map_scale),
        [],
    )


def _rgb(value: Color | list[int]) -> tuple[int, int, int]:
    return (int(value[0]), int(value[1]), int(value[2]))


def default_room_color(number: int | str) -> tuple[int, int, int]:
    """Color the stock palette uses for a room (what the core map shows)."""
    color = ColorsPalette.ROOM_COLORS.get(str(number))
    return _rgb(color) if color is not None else (133, 193, 233)


def default_path_color() -> tuple[int, int, int]:
    """Color the stock palette uses for the vacuum path."""
    return _rgb(ColorsPalette.COLORS[SupportedColor.PATH])


def colors_palette(
    record: dict,
    *,
    show_background: bool = False,
    show_walls: bool = True,
    show_rooms: bool = True,
    extra: dict[SupportedColor, Color] | None = None,
) -> ColorsPalette:
    """Palette of the core integration with the user's room/path colors on top.

    The show_* flags mirror the core Roborock options (same defaults), so a map
    re-rendered here looks exactly like the core one apart from the custom colors.
    """
    colors: dict[SupportedColor, Color] = {}
    rooms: dict[str, Color] = {}
    if not show_background:
        colors[SupportedColor.MAP_OUTSIDE] = _TRANSPARENT
    if not show_walls:
        for wall_color in _WALL_COLORS:
            colors[wall_color] = _TRANSPARENT
    if not show_rooms:
        rooms = {str(room_id): _TRANSPARENT for room_id in range(1, 32)}

    if (path := record.get(CONF_PATH_COLOR)) is not None:
        colors[SupportedColor.PATH] = _rgb(path)
    for number, color in record.get(CONF_ROOM_COLORS, {}).items():
        rooms[str(number)] = _rgb(color)
    if extra:
        colors.update(extra)
    return ColorsPalette(colors, rooms)


# Submap bits (one byte per map pixel). 0x01 is carpet; 0x04 (unknown to the
# parser) marks the area covered by the last clean on Saros models, which the
# Roborock app shows as a lighter band around the path.
_SUBMAP_CARPET_BITS = 0x01
_SUBMAP_CLEANED_BITS = 0x04
_CLEANED_LIGHTEN = 0.28
# Width of the mopped band drawn along the mop path, in map pixels (5 cm each).
_MOP_BAND_CELLS = 6
_SUBMAP_REMOVED_BITS = 0x08 | 0x10
_BLOCK_IMAGE = 2
_BLOCK_SUBMAP = 17
_CARPET_SHADE = 0.8


def _int16(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 2], "little")


def _int32(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 4], "little", signed=True)


def carpet_cells(
    raw: bytes, bits: int = _SUBMAP_CARPET_BITS
) -> tuple[int, int, set[tuple[int, int]]] | None:
    """(width, height, cells with any of bits) in image coordinates, or None.

    Cells use the parser's orientation: x = column, y = row counted from the top
    of the rendered image (the raw rows are stored bottom-up).
    """
    pos = _int16(raw, 0x02)
    width = height = None
    submap: bytes | None = None
    while pos + 8 <= len(raw):
        header_length = _int16(raw, pos + 0x02)
        block_type = _int16(raw, pos)
        data_length = _int32(raw, pos + 0x04)
        if header_length <= 0 or data_length < 0:
            break
        data_start = pos + header_length
        if block_type == _BLOCK_IMAGE:
            header = raw[pos : pos + header_length]
            height = _int32(header, header_length - 8)
            width = _int32(header, header_length - 4)
        elif block_type == _BLOCK_SUBMAP:
            submap = raw[data_start : data_start + data_length]
        pos = data_start + data_length
    if not width or not height or not submap:
        return None
    cells: set[tuple[int, int]] = set()
    for idx, value in enumerate(submap[: width * height]):
        if value & bits and not value & _SUBMAP_REMOVED_BITS:
            cells.add((idx % width, height - 1 - idx // width))
    return width, height, cells


def paint_carpets(
    image: Image.Image, raw: bytes, map_scale: int, shade: float = _CARPET_SHADE
) -> Image.Image:
    """Draw carpets like the Roborock app: a checkerboard of the room color, darker.

    Only cells on the "odd" squares are darkened (same pattern the parser uses);
    their color is taken from the neighbouring square, so it follows the room
    color and also replaces the parser's own green carpet squares. shade=1.0
    hides carpets entirely.
    """
    found = carpet_cells(raw)
    if found is None:
        return image
    width, height, cells = found
    if not cells or image.size != (width * map_scale, height * map_scale):
        return image
    image = image.convert("RGBA")
    px = image.load()
    half = map_scale // 2
    for x, y in cells:
        if not (x + y) % 2:
            continue
        base = None
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height:
                color = px[nx * map_scale + half, ny * map_scale + half]
                if color[3] > 0:
                    base = color
                    break
        if base is None:
            continue
        shaded = (
            int(base[0] * shade),
            int(base[1] * shade),
            int(base[2] * shade),
            base[3],
        )
        x0, y0 = x * map_scale, y * map_scale
        for dy in range(map_scale):
            for dx in range(map_scale):
                px[x0 + dx, y0 + dy] = shaded
    return image


def paint_cleaned(
    image: Image.Image, raw: bytes, map_scale: int, amount: float = _CLEANED_LIGHTEN
) -> Image.Image:
    """Fallback when the map has no mop path: lighten the cells covered by the
    last clean, except carpets (the mop is lifted there)."""
    found = carpet_cells(raw, _SUBMAP_CLEANED_BITS)
    if found is None:
        return image
    width, height, cells = found
    carpets = carpet_cells(raw)
    if carpets is not None:
        cells -= carpets[2]  # the mop is lifted on carpets
    if not cells or image.size != (width * map_scale, height * map_scale):
        return image
    image = image.convert("RGBA")
    px = image.load()
    half = map_scale // 2
    walls = {_rgb(ColorsPalette.COLORS[c]) for c in _WALL_COLORS}
    for x, y in cells:
        base = px[x * map_scale + half, y * map_scale + half]
        if base[3] == 0 or base[:3] in walls:
            continue
        light = (
            int(base[0] + (255 - base[0]) * amount),
            int(base[1] + (255 - base[1]) * amount),
            int(base[2] + (255 - base[2]) * amount),
            base[3],
        )
        x0, y0 = x * map_scale, y * map_scale
        for dy in range(map_scale):
            for dx in range(map_scale):
                px[x0 + dx, y0 + dy] = light
    return image


# Map pixels the parser can't classify (obstacle codes 2-6) are drawn in the
# UNKNOWN color, black by default. The app shows them as a darker room color.
_UNKNOWN_SENTINEL: Color = (1, 2, 3, 255)
_UNKNOWN_SHADE = 0.72
_UNKNOWN_SEARCH = 4


def paint_unknown(image: Image.Image, map_scale: int) -> Image.Image:
    """Replace sentinel (unknown) cells with a darker shade of the nearest room color."""
    image = image.convert("RGBA")
    px = image.load()
    width, height = image.width // map_scale, image.height // map_scale
    half = map_scale // 2
    sentinel = _UNKNOWN_SENTINEL[:3]
    walls = {
        _rgb(ColorsPalette.COLORS[c])
        for c in (*_WALL_COLORS, SupportedColor.MAP_INSIDE, SupportedColor.SCAN)
    }

    def cell(x: int, y: int) -> tuple[int, int, int, int]:
        return px[x * map_scale + half, y * map_scale + half]

    targets = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if cell(x, y)[:3] == sentinel
    ]
    fallback = _rgb(ColorsPalette.COLORS[SupportedColor.GREY_WALL]) + (255,)
    for x, y in targets:
        found = None
        for radius in range(1, _UNKNOWN_SEARCH + 1):
            for dx, dy in (
                (radius, 0), (-radius, 0), (0, radius), (0, -radius),
                (radius, radius), (-radius, -radius), (radius, -radius), (-radius, radius),
            ):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    color = cell(nx, ny)
                    if color[3] > 0 and color[:3] != sentinel and color[:3] not in walls:
                        found = color
                        break
            if found is not None:
                break
        if found is None:
            shaded = fallback
        else:
            shaded = (
                int(found[0] * _UNKNOWN_SHADE),
                int(found[1] * _UNKNOWN_SHADE),
                int(found[2] * _UNKNOWN_SHADE),
                found[3],
            )
        x0, y0 = x * map_scale, y * map_scale
        for oy in range(map_scale):
            for ox in range(map_scale):
                px[x0 + ox, y0 + oy] = shaded
    return image


def build_colored_map(
    raw: bytes,
    drawables: list[Drawable],
    map_scale: int,
    record: dict,
    core_flags: dict[str, bool] | None = None,
) -> bytes:
    """Re-render the raw map like the core integration, but with custom colors.

    Uses the same parser settings as the core Roborock integration, so the image
    size (and therefore the calibration points) stays identical. The floor is
    rendered first (unknown cells shaded like the app, carpets as a darker
    checkerboard when enabled), then the drawables (path, vacuum, charger...)
    are composited on top.
    """
    palette = colors_palette(
        record, **(core_flags or {}), extra={SupportedColor.UNKNOWN: _UNKNOWN_SENTINEL}
    )
    floor = _create_parser(palette, [], map_scale).parse(raw)
    if floor.image is None:
        raise ValueError("Raw map data could not be rendered")
    image = paint_unknown(floor.image.data, map_scale)
    show_mopped = record.get(CONF_CLEANED_AREA, True)
    has_mop_path = bool(floor.mop_path and floor.mop_path.path)
    if show_mopped and not has_mop_path:
        image = paint_cleaned(image, raw, map_scale)
    carpets_app = record.get(CONF_CARPETS, CARPETS_APP) == CARPETS_APP
    image = paint_carpets(image, raw, map_scale, _CARPET_SHADE if carpets_app else 1.0)
    if show_mopped and has_mop_path:
        # Like the app: a translucent band along the path where the mop was down,
        # drawn below the path itself.
        mopped = _create_parser(
            _overlay_palette(), [Drawable.MOP_PATH], map_scale,
            mop_width=_MOP_BAND_CELLS * map_scale,
        ).parse(raw)
        if mopped.image is not None and mopped.image.data.size == image.size:
            image.alpha_composite(mopped.image.data.convert("RGBA"))
    drawables = [d for d in drawables if d != Drawable.MOP_PATH]
    overlay = build_overlay(raw, drawables, map_scale, record.get(CONF_PATH_COLOR))
    if overlay.size != image.size:
        raise ValueError("Overlay and floor sizes differ")
    image.alpha_composite(overlay)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def build_overlay(
    raw: bytes,
    drawables: list[Drawable],
    map_scale: int,
    path_color: Color | list[int] | None = None,
) -> Image.Image:
    """Parse the raw map into an RGBA layer containing only the drawables.

    path_color is the user's custom vacuum path color (if any), so it also
    applies on top of a Custom Floor Plan.
    """
    overlay_map = _create_parser(
        _overlay_palette(path_color), drawables, map_scale
    ).parse(raw)
    if overlay_map.image is None:
        raise ValueError("Raw map data could not be rendered")
    return overlay_map.image.data.convert("RGBA")


def build_wall_mask(raw: bytes, map_scale: int) -> Image.Image:
    """Parse the raw map into a single-band alpha mask of walls and charger."""
    mask_map = _create_parser(_mask_palette(), [Drawable.CHARGER], map_scale).parse(raw)
    if mask_map.image is None:
        raise ValueError("Raw map data could not be rendered")
    return mask_map.image.data.convert("RGBA").getchannel("A")


@lru_cache(maxsize=4)
def _scaled_background(bg_bytes: bytes, scale_x: float, scale_y: float) -> Image.Image:
    """Decode and scale the background image (cached; treat the result read-only).

    Scales are reduced so the result never exceeds MAX_SCALED_DIMENSION per axis
    or MAX_SCALED_PIXELS in total.
    """
    with Image.open(io.BytesIO(bg_bytes)) as src:
        bg = ImageOps.exif_transpose(src).convert("RGBA")
    scale_x = min(scale_x, MAX_SCALED_DIMENSION / bg.width)
    scale_y = min(scale_y, MAX_SCALED_DIMENSION / bg.height)
    if (area := bg.width * scale_x * bg.height * scale_y) > MAX_SCALED_PIXELS:
        shrink = math.sqrt(MAX_SCALED_PIXELS / area)
        scale_x *= shrink
        scale_y *= shrink
    if scale_x != 1 or scale_y != 1:
        bg = bg.resize(
            (max(1, round(bg.width * scale_x)), max(1, round(bg.height * scale_y))),
            Image.Resampling.LANCZOS,
        )
    return bg


def clear_scaled_backgrounds() -> None:
    """Drop cached decoded backgrounds (call on unload and when flows end)."""
    _scaled_background.cache_clear()


_ROTATE_TRANSPOSE = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


@dataclass(frozen=True)
class MapTransform:
    """Affine map between vacuum-world mm and base (unrotated) render pixels."""

    dims: ImageDimensions

    @classmethod
    def from_dimensions(cls, dims: ImageDimensions) -> MapTransform:
        return cls(dims)

    @property
    def base_size(self) -> tuple[int, int]:
        """Unrotated base render size in px (equals the parsed overlay size)."""
        return (
            int(self.dims.width * self.dims.scale),
            int(self.dims.height * self.dims.scale),
        )

    def vacuum_to_base(self, wx: float, wy: float) -> tuple[float, float]:
        """Vacuum-world mm -> base (unrotated) render pixel."""
        p = self.dims.to_img(Point(wx, wy))
        return (p.x, p.y)

    def base_to_vacuum(self, px: float, py: float) -> tuple[float, float]:
        """Base (unrotated) render pixel -> vacuum-world mm."""
        dims = self.dims
        return (
            (px / dims.scale + dims.left) * 50,
            (dims.top + dims.height - 1 - py / dims.scale) * 50,
        )


def rotate_point(
    x: float, y: float, w: int, h: int, rotation: int
) -> tuple[float, float]:
    """Rotate a base-frame pixel (w x h) counter-clockwise into the rotated frame."""
    if rotation == 90:
        return (y, w - x)
    if rotation == 180:
        return (w - x, h - y)
    if rotation == 270:
        return (h - y, x)
    return (x, y)


def unrotate_point(
    x: float, y: float, w: int, h: int, rotation: int
) -> tuple[float, float]:
    """Inverse of rotate_point; (w, h) is the UNROTATED base size (not swapped)."""
    if rotation == 90:
        return (w - y, x)
    if rotation == 180:
        return (w - x, h - y)
    if rotation == 270:
        return (y, h - x)
    return (x, y)


def anchor_topleft(
    transform: MapTransform, anchor_vx: float, anchor_vy: float, rotation: int
) -> tuple[float, float]:
    """Rotated-frame top-left px of an override anchored in vacuum-world mm."""
    bx, by = transform.vacuum_to_base(anchor_vx, anchor_vy)
    w_px, h_px = transform.base_size
    return rotate_point(bx, by, w_px, h_px, rotation)


def canvas_origin(offset_x: float, offset_y: float) -> tuple[int, int]:
    """Top-left of the union canvas (<= 0 per axis), a function of the offset alone."""
    return (math.floor(min(0.0, offset_x)), math.floor(min(0.0, offset_y)))


def union_canvas(
    rot_w: int,
    rot_h: int,
    offset_x: float,
    offset_y: float,
    ovr_w: int,
    ovr_h: int,
) -> tuple[int, int, int, int]:
    """Union bbox of the rotated base extent and the override extent.

    Returns (canvas_w, canvas_h, origin_x, origin_y). The size is capped per axis
    and in total pixels with the origin preserved, so content may be cropped at
    the right/bottom.
    """
    origin_x, origin_y = canvas_origin(offset_x, offset_y)
    canvas_w = math.ceil(max(rot_w, offset_x + ovr_w)) - origin_x
    canvas_h = math.ceil(max(rot_h, offset_y + ovr_h)) - origin_y
    canvas_w = min(canvas_w, MAX_COMPOSITE_DIMENSION)
    canvas_h = min(canvas_h, MAX_COMPOSITE_DIMENSION)
    if canvas_w * canvas_h > MAX_COMPOSITE_PIXELS:
        shrink = math.sqrt(MAX_COMPOSITE_PIXELS / (canvas_w * canvas_h))
        canvas_w = max(1, math.floor(canvas_w * shrink))
        canvas_h = max(1, math.floor(canvas_h * shrink))
    return (canvas_w, canvas_h, origin_x, origin_y)


def compose_final(
    overlay: Image.Image,
    mask: Image.Image | None,
    bg_bytes: bytes,
    offset_x: float,
    offset_y: float,
    scale_x: float,
    scale_y: float,
    rotation: int,
) -> bytes:
    """Composite the background under the drawables and encode as PNG.

    offset_x/offset_y are the scaled override's top-left in the rotated frame.
    """
    if rotation in _ROTATE_TRANSPOSE:
        overlay = overlay.transpose(_ROTATE_TRANSPOSE[rotation])
        if mask is not None:
            mask = mask.transpose(_ROTATE_TRANSPOSE[rotation])
    rot_w, rot_h = overlay.size

    bg = _scaled_background(bg_bytes, scale_x, scale_y)
    ovr_w, ovr_h = bg.size

    canvas_w, canvas_h, origin_x, origin_y = union_canvas(
        rot_w, rot_h, offset_x, offset_y, ovr_w, ovr_h
    )

    canvas = Image.new("RGBA", (canvas_w, canvas_h), _TRANSPARENT)
    bg_dest = (round(offset_x - origin_x), round(offset_y - origin_y))
    if bg_dest[0] < canvas_w and bg_dest[1] < canvas_h:
        canvas.alpha_composite(bg, dest=bg_dest)

    base_dest = (-origin_x, -origin_y)
    if base_dest[0] < canvas_w and base_dest[1] < canvas_h:
        canvas.alpha_composite(overlay, dest=base_dest)
        if mask is not None:
            full_mask = Image.new("L", (canvas_w, canvas_h), 0)
            full_mask.paste(mask, base_dest)
            inverted = ImageChops.invert(canvas.convert("RGB")).convert("RGBA")
            canvas = Image.composite(inverted, canvas, full_mask)

    out = io.BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()


def validate_upload(data: bytes) -> str:
    """Validate an uploaded background image and return its file extension."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidBackgroundImage("file_too_large")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").lower()
            width, height = image.size
            image.verify()
    except Image.DecompressionBombError as err:
        raise InvalidBackgroundImage("image_too_large") from err
    except Exception as err:
        raise InvalidBackgroundImage("invalid_image") from err
    if image_format not in ALLOWED_FORMATS:
        raise InvalidBackgroundImage("invalid_image")
    if width > MAX_BG_DIMENSION or height > MAX_BG_DIMENSION:
        raise InvalidBackgroundImage("image_too_large")
    return ALLOWED_FORMATS[image_format]


def override_dir(hass: HomeAssistant) -> Path:
    """Directory where uploaded background files are stored."""
    return Path(hass.config.path(STORAGE_DIR, DOMAIN))


def save_override_file(hass: HomeAssistant, filename: str, data: bytes) -> None:
    """Persist an uploaded background image."""
    directory = override_dir(hass)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_bytes(data)


def read_override_file(hass: HomeAssistant, filename: str) -> bytes:
    """Read a stored background image (raises OSError when missing)."""
    return (override_dir(hass) / filename).read_bytes()


def delete_override_files_for_key(hass: HomeAssistant, rotation_key: str) -> None:
    """Delete stored background files for a map (any extension)."""
    directory = override_dir(hass)
    if not directory.is_dir():
        return
    for path in directory.glob(f"{rotation_key}.*"):
        path.unlink(missing_ok=True)

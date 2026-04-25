from __future__ import annotations

import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "assets" / "reward_curve.png"


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def line(canvas: list[list[tuple[int, int, int]]], x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                x, y = x0 + ox, y0 + oy
                if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
                    canvas[y][x] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    raw = b"".join(b"\x00" + b"".join(bytes(rgb) for rgb in row) for row in pixels)
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw, 9))
    data += chunk(b"IEND", b"")
    path.write_bytes(data)


def main() -> None:
    width, height = 900, 500
    bg = (250, 250, 248)
    ink = (37, 43, 54)
    weak = (188, 83, 75)
    trained = (39, 128, 113)
    grid = (218, 222, 226)
    pixels = [[bg for _ in range(width)] for _ in range(height)]
    left, right, top, bottom = 90, 840, 52, 420
    for i in range(6):
        y = top + int((bottom - top) * i / 5)
        line(pixels, left, y, right, y, grid)
    line(pixels, left, top, left, bottom, ink)
    line(pixels, left, bottom, right, bottom, ink)
    weak_series = [0.18, 0.22, 0.20, 0.24, 0.25, 0.27, 0.25, 0.26]
    trained_series = [0.22, 0.32, 0.41, 0.53, 0.61, 0.70, 0.78, 0.84]
    for series, color in [(weak_series, weak), (trained_series, trained)]:
        pts = []
        for i, value in enumerate(series):
            x = left + int((right - left) * i / (len(series) - 1))
            y = bottom - int((bottom - top) * value)
            pts.append((x, y))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            line(pixels, x0, y0, x1, y1, color)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    write_png(PNG, pixels)


if __name__ == "__main__":
    main()

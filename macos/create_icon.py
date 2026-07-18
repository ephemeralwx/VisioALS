#!/usr/bin/env python3
"""Generate the VisioALS.icns application icon using Pillow."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


CANVAS_SIZE = 1024
def make_master() -> Image.Image:
    image = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    pixels = image.load()

    # A restrained teal-to-indigo field matches the app's accessibility focus
    # while remaining legible at Dock and Finder sizes.
    top = (30, 116, 128)
    bottom = (55, 54, 105)
    for y in range(CANVAS_SIZE):
        amount = y / (CANVAS_SIZE - 1)
        color = tuple(round(a + (b - a) * amount) for a, b in zip(top, bottom))
        for x in range(CANVAS_SIZE):
            pixels[x, y] = (*color, 255)

    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (38, 38, 986, 986), radius=218, fill=255
    )
    image.putalpha(mask)
    draw = ImageDraw.Draw(image)

    # Eye mark.
    draw.ellipse((158, 222, 866, 692), fill=(247, 250, 246, 248))
    draw.ellipse((348, 260, 676, 650), fill=(72, 168, 157, 255))
    draw.ellipse((416, 326, 608, 584), fill=(29, 40, 61, 255))
    draw.ellipse((462, 362, 520, 438), fill=(255, 255, 255, 235))

    # A small speech tail turns the eye into a communication bubble.
    draw.polygon(
        ((664, 634), (812, 776), (746, 612)),
        fill=(247, 250, 246, 248),
    )

    # Three response dots remain recognizable in small Finder views.
    for center_x in (402, 512, 622):
        draw.ellipse(
            (center_x - 28, 740, center_x + 28, 796),
            fill=(235, 244, 239, 235),
        )
    return image


def create_icns(output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    master = make_master()
    master.save(output, format="ICNS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    create_icns(args.output)


if __name__ == "__main__":
    main()

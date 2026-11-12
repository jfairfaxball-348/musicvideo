"""Animate a unique diffusion plate when both CUDA and CPU motion renders fail."""

import math
import subprocess
import numpy as np
from PIL import Image, ImageFilter


def photographic_insert(ff, plate, dest, frames, shot, log):
    image = (
        Image.open(plate).convert("RGB").resize((768, 432), Image.Resampling.LANCZOS)
    )
    width, height = image.size
    rng = np.random.default_rng(shot["seed"])
    fog = (
        Image.fromarray(rng.integers(0, 255, (18, 40), dtype=np.uint8))
        .resize((width + 100, height), Image.Resampling.BICUBIC)
        .filter(ImageFilter.GaussianBlur(9))
    )
    tint = Image.new("RGB", image.size, (212, 174, 123))
    direction = 1 if shot["seed"] % 2 else -1

    def sample(x, y, t):
        nx, ny = x / width, y / height
        if shot["subject"] in ("stage", "crowd"):
            depth = ny * ny
        else:
            depth = math.exp(-((nx - 0.49) ** 2 / 0.064 + (ny - 0.52) ** 2 / 0.22))
        zoom = 1.07 + 0.025 * t
        return (
            (x - width / 2) / zoom + width / 2 + direction * t * (5 + 11 * depth),
            (y - height / 2) / zoom + height / 2 - 2 * t * depth,
        )

    cmd = [
        ff,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "768x432",
        "-framerate",
        "24",
        "-i",
        "-",
        "-vf",
        "scale=1920:1080:flags=lanczos,noise=alls=2:allf=t+u",
        "-frames:v",
        str(frames),
        "-c:v",
        "libx264",
        "-threads",
        "5",
        "-bf",
        "0",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    with (
        open(log, "ab") as out,
        subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=out) as encoder,
    ):
        for i in range(frames):
            t = i / max(1, frames - 1) - 0.5
            mesh = []
            for top in range(0, height, 54):
                for left in range(0, width, 64):
                    right, bottom = left + 64, top + 54
                    quad = sum(
                        (
                            sample(x, y, t)
                            for x, y in (
                                (left, top),
                                (left, bottom),
                                (right, bottom),
                                (right, top),
                            )
                        ),
                        (),
                    )
                    mesh.append(((left, top, right, bottom), quad))
            frame = image.transform(
                image.size,
                Image.Transform.MESH,
                mesh,
                resample=Image.Resampling.BICUBIC,
            )
            shift = int(50 + 60 * t)
            veil = fog.crop((shift, 0, shift + width, height)).point(
                lambda v: int(v * 0.055)
            )
            encoder.stdin.write(Image.composite(tint, frame, veil).tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Photographic insert encode failed")

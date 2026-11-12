"""Serial, restartable local concert renderer. Run from the repository root."""

import hashlib, json, math, os, subprocess, time, msvcrt
import av
import numpy as np
from PIL import Image
from imageio_ffmpeg import get_ffmpeg_exe
from analyse import ROOT, PLAN, FPS, source, write

LOCAL = ROOT / ".local"
OUT = ROOT / "output"
REF = LOCAL / "refs"
LOG = LOCAL / "logs"
CLIPS = LOCAL / "shots"
SD = LOCAL / "bin/sd-cli.exe"
FF = get_ffmpeg_exe()
NEG = "animation, illustration, digital painting, plastic skin, deformed anatomy, extra limbs, warped instruments, captions, logos, text, watermark, static image, frozen crowd, modern smartphones"
PROFILES = [(768, 432, 22, 16), (640, 368, 20, 12), (512, 288, 20, 10)]
FINAL = OUT / "LNMA_Live_Rap_Metal_Official_Music_Video.mp4"


def run(args, log):
    with (LOG / log).open("ab") as f:
        f.write(("\n" + subprocess.list2cmdline(list(map(str, args))) + "\n").encode())
        f.flush()
        subprocess.run(
            list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, check=True
        )


def generate(path, prompt, seed, n, w, h, steps, ref=None, cpu=False):
    models = LOCAL / "models"
    args = [
        SD,
        "-M",
        "vid_gen",
        "--diffusion-model",
        models / "wan2.1-vace-1.3b-q5_k_m.gguf",
        "-m",
        models / "embed.safetensors",
        "--vae",
        models / "wan_2.1_vae.safetensors",
        "--t5xxl",
        models / "umt5-xxl-encoder-Q4_K_M.gguf",
        "--backend",
        "cpu" if cpu else "te=cpu,diffusion=cuda0,vae=cuda0",
        "--params-backend",
        "cpu",
        "--flash-attn",
        "on",
        "--vae-tiling",
        "--vae-tile-size",
        "256",
        "--temporal-tiling",
        "--steps",
        steps,
        "--cfg-scale",
        "5",
        "--sampling-method",
        "euler",
        "--flow-shift",
        "4",
        "-W",
        w,
        "-H",
        h,
        "--video-frames",
        n,
        "--fps",
        16,
        "-s",
        seed,
        "-p",
        prompt,
        "-n",
        NEG,
        "-o",
        path,
    ]
    if ref:
        args += ["-i", ref]
    run(args, path.stem + ".log")


def valid_ref(path):
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("L")).std() > 8
    except (OSError, ValueError):
        return False


def ref_for(subject, identity):
    path = REF / (subject + ".png")
    if valid_ref(path):
        return path
    tmp = REF / (subject + "-new.png")
    prompt = (
        identity["cast"][subject]
        + " "
        + identity["look"]
        + " Candid medium distance concert press photograph with clear subject and stage environment."
    )
    for w, h, cpu in [(768, 432, False), (640, 368, False), (640, 368, True)]:
        try:
            generate(tmp, prompt, identity["base_seeds"][subject], 1, w, h, 26, cpu=cpu)
            if valid_ref(tmp):
                tmp.replace(path)
                return path
        except subprocess.CalledProcessError:
            pass
    raise RuntimeError("Failed reference generation: " + subject)


def inspect(path, n, size=None):
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        if size and (stream.width, stream.height) != size:
            raise ValueError("dimensions")
        prev = None
        motion = []
        frames = 0
        black = 0
        for f in c.decode(stream):
            thumb = (
                f.reformat(width=128, height=72, format="gray")
                .to_ndarray()
                .astype(float)
            )
            if thumb.std() < 4 or thumb.mean() < 5:
                black += 1
            if prev is not None:
                motion.append(float(np.abs(thumb - prev).mean()))
            prev = thumb
            frames += 1
        if frames != n or black or (motion and max(motion) < 0.55):
            raise ValueError(
                f"Bad clip {path}: frames={frames}/{n}, dark={black}, motion={max(motion, default=0)}"
            )
        return round(float(np.mean(motion)), 3) if motion else 0


def accent_timing(raw, s, gen, n):
    """Move one measured motion peak onto a nearby song beat with a mild retime."""
    base = f"N*{n}/({gen}*{FPS}*TB)"
    if s["energy"] < 0.45 or gen < 13:
        return base, None
    samples = []
    with av.open(str(raw)) as clip:
        for f in clip.decode(video=0):
            samples.append(f.reformat(width=128, height=72, format="gray").to_ndarray())
    motion = np.abs(np.diff(np.stack(samples).astype(float), axis=0)).mean(axis=(1, 2))
    motion = np.convolve(motion, [0.25, 0.5, 0.25], mode="same")
    lo, hi = gen // 4, 3 * gen // 4
    peak = lo + int(np.argmax(motion[lo:hi])) + 1
    beats = json.loads((PLAN / "timeline.json").read_text())["beats"]
    dur = n / FPS
    natural = peak * dur / gen
    interior = [b - s["start"] for b in beats if s["start"] + 0.2 < b < s["end"] - 0.2]
    if not interior:
        return base, None
    target = min(interior, key=lambda b: abs(b - natural))
    if abs(target - natural) > 0.18 * min(natural, dur - natural):
        return base, None
    # Quotes protect the filter expression's commas in the FFmpeg graph.
    expr = f"'if(lte(N,{peak}),N*{target}/{peak},{target}+(N-{peak})*{dur - target}/{gen - peak})/TB'"
    return expr, round(s["start"] + target, 4)


def render_shot(s, ref, path):
    n = s["end_frame"] - s["start_frame"]
    raw = CLIPS / (path.stem + "-raw.avi")
    temp = CLIPS / (path.stem + "-new.mp4")
    for i, (w, h, steps, rate) in enumerate([*PROFILES, (512, 288, 20, 8)]):
        cpu = i == len(PROFILES)
        gen = 4 * math.ceil((n * rate / FPS - 1) / 4) + 1
        if cpu:
            gen = min(25, gen)
        # The 57-frame 768-wide pilot exhausted this laptop's device allocator.
        if w == 768 and gen >= 57:
            continue
        try:
            generate(raw, s["prompt"], s["seed"] + i, gen, w, h, steps, ref, cpu=cpu)
            inspect(raw, gen)
            timing, accent = accent_timing(raw, s, gen, n)
            # Give each diffusion frame its intended screen time, interpolate at acquisition size.
            vf = f"settb=1/96000,setpts={timing},tpad=stop_duration=0.2:stop_mode=clone,minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,tpad=stop_duration=0.2:stop_mode=clone,scale=1920:1080:flags=lanczos,setsar=1,eq=contrast=1.035:saturation=0.91,noise=alls=2:allf=t+u"
            run(
                [
                    FF,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    raw,
                    "-vf",
                    vf,
                    "-frames:v",
                    n,
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "5",
                    "-crf",
                    "17",
                    "-preset",
                    "fast",
                    "-bf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    temp,
                ],
                path.stem + ".log",
            )
            motion = inspect(temp, n, (1920, 1080))
            temp.replace(path)
            raw.unlink(missing_ok=True)
            return dict(
                width=w,
                height=h,
                steps=steps,
                seed=s["seed"] + i,
                generated_frames=gen,
                motion=motion,
                accent_beat=accent,
                backend="CPU" if cpu else "CUDA",
                technique="reference-to-video",
            )
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            print("Retry", path.stem, i + 1, str(e), flush=True)
    # A unique reference-conditioned photograph per failed shot, atmospherics and optical drift.
    still = CLIPS / (path.stem + "-plate.png")
    generate(still, s["prompt"], s["seed"], 1, 640, 368, 26, ref)
    if not valid_ref(still):
        raise ValueError(still)
    from insert import photographic_insert

    photographic_insert(FF, still, temp, n, s, LOG / (path.stem + ".log"))
    motion = inspect(temp, n, (1920, 1080))
    temp.replace(path)
    return dict(
        technique="moving photographic insert", seed=s["seed"], steps=26, motion=motion
    )


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    os.chdir(ROOT)
    for d in (OUT, REF, LOG, CLIPS):
        d.mkdir(exist_ok=True, parents=True)
    with (LOCAL / "worker.lock").open("ab+") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        shots = json.loads((PLAN / "shots.json").read_text())
        identity = json.loads((PLAN / "identity.json").read_text())
        timeline = json.loads((PLAN / "timeline.json").read_text())
        if digest(source()) != timeline["audio_sha256"]:
            raise ValueError("Soundtrack changed; rebuild timeline")
        manifest = OUT / "manifest.json"
        state = (
            json.loads(manifest.read_text()) if manifest.exists() else {"completed": {}}
        )
        state.update(total=len(shots), status="rendering", pid=os.getpid())
        write(manifest, state)
        for s in shots:
            if (LOCAL / "STOP").exists():
                state["status"] = "paused"
                write(manifest, state)
                return
            key = f"{s['id']:04d}"
            path = CLIPS / (key + ".mp4")
            spec = hashlib.sha256(
                json.dumps([s, identity, PROFILES]).encode()
            ).hexdigest()
            last = state["completed"].get(key, {})
            if (
                path.exists()
                and last.get("spec") == spec
                and last.get("sha256") == digest(path)
            ):
                continue
            state["current"] = key
            write(manifest, state)
            start = time.monotonic()
            print("Rendering", key, "of", len(shots), s["subject"], flush=True)
            ref = ref_for(s["subject"], identity)
            result = render_shot(s, ref, path)
            state["completed"][key] = dict(
                **result,
                seconds=round(time.monotonic() - start, 1),
                spec=spec,
                sha256=digest(path),
            )
            state["remaining"] = len(shots) - len(state["completed"])
            write(manifest, state)
        from finish import assemble

        state["status"] = "assembling"
        write(manifest, state)
        assemble(shots, timeline)
        state = json.loads(manifest.read_text())
        state["status"] = "validated"
        write(manifest, state)


if __name__ == "__main__":
    main()

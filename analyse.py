from pathlib import Path
import json, hashlib, math
import numpy as np
import soundfile as sf
import librosa
import sherpa_onnx as so
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "production"
FPS = 24


def source():
    files = [
        p
        for p in ROOT.rglob("*.mp3")
        if not any(
            part.startswith(".") or part == "output"
            for part in p.relative_to(ROOT).parts[:-1]
        )
    ]
    if len(files) != 1:
        raise ValueError(f"Expected one soundtrack, found {len(files)}")
    return files[0]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf8")
    temp.replace(path)


def enrich(t):
    """Describe contiguous vocal passages, section transitions and the local pulse."""
    from scipy.ndimage import median_filter

    features = t["half_second_features"]
    active = (
        median_filter(np.array([f["vocal_ratio"] for f in features]), size=5) > 0.23
    )
    starts = np.r_[0, np.flatnonzero(np.diff(active)) + 1]
    t["passages"] = [
        dict(
            start=float(a * 0.5),
            end=min(float(b * 0.5), t["duration"]),
            vocals_likely=bool(active[a]),
        )
        for a, b in zip(starts, np.r_[starts[1:], len(features)])
    ]
    beats = np.asarray(t["beats"])
    for i, sec in enumerate(t["sections"]):
        local = beats[(beats >= sec["start"]) & (beats < sec["end"])]
        if len(local) > 2:
            sec["local_bpm"] = round(60 * (len(local) - 1) / (local[-1] - local[0]), 2)
        sec["energy_change"] = round(
            sec["energy"] - (t["sections"][i - 1]["energy"] if i else 0), 3
        )
    return t


def stem_energy(y, sr):
    folder = ROOT / ".local/sherpa-onnx-spleeter-2stems"
    config = so.OfflineSourceSeparationConfig(
        model=so.OfflineSourceSeparationModelConfig(
            spleeter=so.OfflineSourceSeparationSpleeterModelConfig(
                vocals=str(folder / "vocals.onnx"),
                accompaniment=str(folder / "accompaniment.onnx"),
            ),
            num_threads=3,
            provider="cpu",
        )
    )
    sp = so.OfflineSourceSeparation(config)
    vals = []
    for a in range(0, len(y), sr * 8):
        left = max(0, a - sr)
        chunk = np.ascontiguousarray(y[left : a + sr * 9].T)
        result = sp.process(sample_rate=sr, samples=chunk)
        vocals = np.asarray(result.stems[0].data)[
            :, a - left : a - left + min(sr * 8, len(y) - a)
        ]
        for i in range(0, vocals.shape[1], sr // 2):
            mix = y[a + i : a + i + sr // 2]
            vals.append(
                float(
                    np.sqrt(
                        np.mean(vocals[:, i : i + sr // 2] ** 2)
                        / (np.mean(mix**2) + 1e-9)
                    )
                )
            )
    return np.array(vals)


def main():
    song = source()
    y, sr = sf.read(song, dtype="float32", always_2d=True)
    duration = len(y) / sr
    v = stem_energy(y, sr)
    mono = librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=22050)
    mel = librosa.feature.melspectrogram(y=mono, sr=22050, n_mels=48)
    db = librosa.power_to_db(mel, ref=np.max)
    onset = librosa.onset.onset_strength(S=db, sr=22050)
    tempo, beat = librosa.beat.beat_track(onset_envelope=onset, sr=22050, trim=False)
    beats = librosa.frames_to_time(beat, sr=22050)
    ts = np.arange(0, duration, 0.5)
    sample = np.minimum((ts * 22050 / 512).astype(int), db.shape[1] - 1)
    energy = np.sqrt(
        np.array(
            [np.mean(mono[i : i + 11025] ** 2) for i in range(0, len(mono), 11025)]
        )
    )[: len(ts)]
    energy = np.clip(
        (energy - np.percentile(energy, 5))
        / (np.percentile(energy, 95) - np.percentile(energy, 5) + 1e-8),
        0,
        1,
    )
    v = np.interp(ts, np.arange(len(v)) * 0.5, v)
    feat = np.vstack([db[:, sample] / 20, v * 3, energy * 2])
    smooth = gaussian_filter1d(feat, 3, axis=1)
    change = np.linalg.norm(smooth[:, 8:] - smooth[:, :-8], axis=0)
    peaks = find_peaks(change, distance=14, prominence=np.std(change) * 0.5)[0] + 4

    def snap(t):
        return float(beats[np.argmin(abs(beats - t))]) if len(beats) else t

    edges = [0] + [snap(ts[i]) for i in peaks if 5 < ts[i] < duration - 4] + [duration]
    segments = []
    for a, b in zip(edges, edges[1:]):
        mask = (ts >= a) & (ts < b)
        e = float(energy[mask].mean())
        vocal = float(np.mean(v[mask] > 0.23))
        kind = "instrumental" if vocal < 0.22 else "verse"
        if e > 0.70 and vocal > 0.38:
            kind = "chorus candidate"
        if e < 0.30 and a > 10:
            kind = "breakdown candidate"
        if b == duration:
            kind = "outro"
        if a == 0:
            kind = "opening"
        segments.append(
            dict(
                start=a,
                end=b,
                energy=round(e, 3),
                vocal_fraction=round(vocal, 3),
                section=kind,
            )
        )
    write(
        PLAN / "timeline.json",
        dict(
            audio_sha256=hashlib.sha256(song.read_bytes()).hexdigest(),
            samples=len(y),
            sample_rate=sr,
            duration=duration,
            fps=FPS,
            frames=math.ceil(duration * FPS),
            tempo=float(np.asarray(tempo).item()),
            beats=beats.round(4).tolist(),
            sections=segments,
            peak_time=float(ts[np.argmax(gaussian_filter1d(energy, 3))]),
            half_second_features=[
                dict(
                    t=float(t),
                    energy=round(float(e), 3),
                    vocal_ratio=round(float(w), 3),
                )
                for t, e, w in zip(ts, energy, v)
            ],
        ),
    )


if __name__ == "__main__":
    main()
    path = PLAN / "timeline.json"
    write(path, enrich(json.loads(path.read_text())))

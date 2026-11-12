"""Fetch pinned assets and verify the published SHA256 before installation."""

from pathlib import Path
import subprocess, hashlib, json, urllib.request, zipfile, struct, tarfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / ".local"
TAG = "master-1091-b9d7176"
MODELS = [
    ("calcuis/wan-1.3b-gguf", "wan2.1-vace-1.3b-q5_k_m.gguf"),
    ("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q4_K_M.gguf"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors"),
]


def download(item):
    url, target, digest = item
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with target.open("rb") as f:
            intact = hashlib.file_digest(f, "sha256").hexdigest() == digest
        if not intact:
            target.unlink()
    if not target.exists():
        part = target.with_suffix(target.suffix + ".part")
        subprocess.run(
            ["curl.exe", "-fLsS", "--retry", "5", "-C", "-", "-o", str(part), url],
            check=True,
        )
        with part.open("rb") as f:
            intact = hashlib.file_digest(f, "sha256").hexdigest() == digest
        if not intact:
            part.unlink()
            raise ValueError(target)
        part.replace(target)
    if target.suffix == ".zip":
        with zipfile.ZipFile(target) as z:
            for entry in z.infolist():
                dest = LOCAL / "bin" / entry.filename
                if not dest.exists() or dest.stat().st_size != entry.file_size:
                    z.extract(entry, LOCAL / "bin")
    print(target.name, flush=True)


def main():
    release = json.load(
        urllib.request.urlopen(
            "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/tags/"
            + TAG
        )
    )
    jobs = []
    for a in release["assets"]:
        if a["name"].endswith(("win-cuda12-x64.zip", "win-cu12-x64.zip")):
            jobs.append(
                (
                    a["browser_download_url"],
                    LOCAL / "downloads" / a["name"],
                    a["digest"].split(":")[1],
                )
            )
    lock = ROOT / "models.lock.json"
    if not lock.exists():
        records = []
        for repo, name in MODELS:
            meta = json.load(
                urllib.request.urlopen(
                    "https://huggingface.co/api/models/" + repo + "?blobs=true"
                )
            )
            entry = next(x for x in meta["siblings"] if x["rfilename"] == name)
            records.append(
                dict(
                    repo=repo,
                    name=name,
                    revision=meta["sha"],
                    sha256=entry["lfs"]["sha256"],
                )
            )
        lock.write_text(json.dumps(records, indent=2))
    for a in json.loads(lock.read_text()):
        jobs.append(
            (
                f"https://huggingface.co/{a['repo']}/resolve/{a['revision']}/{a['name']}",
                LOCAL / "models" / Path(a["name"]).name,
                a["sha256"],
            )
        )
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(download, jobs))
    # This GGUF omits one 5D kernel. Load the original tensor alongside it.
    # Only the header and first few MB of the pinned upstream weights are needed.
    modeldir = LOCAL / "models"
    embed = modeldir / "embed.safetensors"
    if not embed.exists():
        upstream = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/123acf1cc74bccbb9bfff8ac1ee72edc08c2341d/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors"
        req = urllib.request.Request(upstream, headers={"Range": "bytes=0-3200000"})
        with urllib.request.urlopen(req) as resp:
            size = struct.unpack("<Q", resp.read(8))[0]
            meta = json.loads(resp.read(size))
            key = "model.diffusion_model.vace_patch_embedding.weight"
            desc = meta[key]
            lo, hi = desc["data_offsets"]
            resp.read(lo)
            data = resp.read(hi - lo)
        desc["data_offsets"] = [0, len(data)]
        header = json.dumps({key: desc}).encode()
        header += b" " * (-len(header) % 8)
        payload = struct.pack("<Q", len(header)) + header + data
        if (
            hashlib.sha256(payload).hexdigest()
            != "25b31d2a0ad628bdfb48f8a45c68392be5385576df4f679ff905ff55c89c2d57"
        ):
            raise ValueError("Unexpected VACE patch embedding")
        embed.write_bytes(payload)
    archive = LOCAL / "spleeter.tar.bz2"
    download(
        (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems.tar.bz2",
            archive,
            "e26401d9c1801f43c0229731d78d32c2e80085e3aceedeb25f29d2de5fa68ca2",
        )
    )
    with tarfile.open(archive) as tar:
        tar.extractall(LOCAL, filter="data")


if __name__ == "__main__":
    main()

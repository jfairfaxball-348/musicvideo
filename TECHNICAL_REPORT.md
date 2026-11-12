# LNMA production record

## Acquisition and analysis

The original stereo MP3 decodes to 14,465,726 samples per channel at 44,100 Hz. The edit spans 328.020997732 seconds, with 7,873 display frames at 24 fps and a shortened final frame. The recording stays in the working directory and its SHA256 is recorded in the timeline. Only the audio features and edit decisions are tracked in Git.

Librosa estimated a pulse of 129.20 BPM and located 705 beats. The segmentation compares smoothed log-mel spectra, vocal activity and energy on either side of each candidate change, then snaps detected transitions to beats. Spleeter runs on CPU in overlapping windows and estimates the vocal-to-mix energy ratio. Verse, chorus and breakdown labels are editorial estimates. The edit contains 21 structural regions and 107 shots. The strongest smoothed energy reading is at 5:09.5.

## Workstation and measured tests

Windows 11 Home, Intel i7-11800H with eight cores and sixteen hardware threads, 15.7 GiB usable system RAM, Intel UHD integrated graphics and NVIDIA RTX 3050 Ti Laptop with 4,096 MiB dedicated VRAM. Before installation the system reported 8.8 GiB available RAM and 35.0 GiB available storage. The NVIDIA driver reported CUDA 13.2 compatibility. The CUDA 12 renderer detected compute capability 8.6 and executed the diffusion and VAE operations on the laptop GPU. The text encoder uses the CPU, parameters reside in host memory and VAE work is spatially tiled. DirectML was unnecessary. A Python 3.12 installation, uv, Git, a cached HTDemucs checkpoint and an imageio FFmpeg binary were present. The project installed its own Python environment and native renderer.

The first VACE GGUF failed during model validation. It lacked the VACE patch-embedding kernel. Loading the original five-dimensional tensor alongside the quantized checkpoint resolved the issue. The first 768 × 432 photographic reference completed the engine's generation phase in 61.58 seconds. Visual inspection showed the first singer had straight wet hair and a vintage microphone. His production reference uses a revised description of the curls, clothing and handheld mic.

## Model and edit choices

Wan 2.1 VACE 1.3B supports reference-conditioned movement, and its upstream model card declares Apache 2.0. The 5-bit diffusion weights and 4-bit text encoder let this production test it on a 4 GiB GPU. The pinned native stable-diffusion.cpp release avoids running a node server or loading a second CUDA framework. Wan 2.2 and ComfyUI are viable candidates on other machines. Mondo provides music-video orchestration but would duplicate the sample-accurate conform and independent Windows worker here. The published Wav2Lip checkpoint restricts commercial use. Vocal shots use profiles, obscuring microphones, over-shoulder views and reaction cuts. Phoneme sync is unverified.

Sources: [Wan VACE model card](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B), [native Wan reference workflows](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/wan.md), [Mondo](https://github.com/hodgemann/mondo-music-video), [ComfyUI Wan examples](https://comfyanonymous.github.io/ComfyUI_examples/wan/), [Wan 2.2](https://github.com/Wan-Video/Wan2.2), [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), [Spleeter ONNX models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models).

Each shot first requests a unique reference-conditioned motion sequence at 768 × 432, nominally 16 generated frames per second. The edit interpolates motion at acquisition size with FFmpeg, upscales with Lanczos, trims by frame number and applies a restrained grain treatment. The opening 57-frame request at 768 pixels wide exhausted the device allocator. The runner skips that configuration for longer shots. Lower-memory CUDA profiles, a short CPU motion profile and unique photographic inserts with depth-dependent movement and drifting haze are available on failure. For high-energy shots, a mild retime can align a measured peak of visual movement with an interior song beat. Actual settings, motion measurements, frame counts, elapsed times and fallback use appear in the local manifest.

## Audio and validation

A direct MP3 stream-copy test produced 1,137 extra decoded samples in an MP4. A 24-bit ALAC encode preserved the decoded soundtrack hash. A conform test with two H.264 segments and a shortened final frame preserved exactly 46,700 audio samples of duration in both track headers after muxing. The release check repeats the comparisons on the actual master, decodes every frame and audio sample, checks frame timestamps, and creates one labeled thumbnail per shot.

Final run evidence is generated in `output/manifest.json`, `output/validation.json` and the contact sheets. Character likeness, hand and instrument geometry, concert continuity, lip articulation and audience reactions require visual review beyond the automated checks.

## First production shot

Shot 0001 completed the 640 × 368 reference-to-video pass with 41 generated frames and 20 sampling steps. The engine reported 527.66 seconds for that pass. The complete shot, including the first reference, the higher-resolution memory failure, retry and encode, took 751.0 seconds. The 1920 × 1080 H.264 shot decoded to 82 display frames. Four inspection frames show the camera advancing toward the outdoor stage between groups of spectators. The camera is in front of the stage despite the rear-view prompt. This shot supplies the establishing view; the remaining shots have their own prompts and seeds.

The independent worker resumed at shot 0002 after a boundary restart and preserved the opening clip hash. The Windows sign-in task was registered and read back as ready. An actual machine reboot was not performed.

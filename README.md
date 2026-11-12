# LNMA live festival film

This Windows production reconstructs an original late-1990s rap-metal concert around the supplied recording. The edit uses separate reference-conditioned diffusion runs for individual shots. All inference, audio analysis, interpolation and encoding run on the local machine.

## Run or resume

From the repository root in PowerShell, with Python 3.12, uv, curl and the NVIDIA driver installed, place one MP3 anywhere in the repository outside generated or hidden directories. Setup resolves the pinned models, installs the native CUDA renderer and rebuilds the edit from the actual song.

```powershell
.\Start-Production.ps1 -Setup
.\Start-Production.ps1 -Status
```

For an existing installation and edit, run the first command below. To stop at the next shot boundary, use the second. The third runs the worker in the current console.

```powershell
.\Start-Production.ps1
.\Start-Production.ps1 -StopAfterShot
.\Start-Production.ps1 -Foreground
```

The background worker owns an exclusive Windows file lock. Its manifest in `output/manifest.json` records shot settings, work times, hashes and progress. Successfully recorded shots survive a reboot. Every restart verifies their hashes. An interrupted shot is regenerated. Generation, retries, interpolation and mux logs are under `.local/logs`; references and encoded shots are under `.local/refs` and `.local/shots`. On the production laptop the Windows scheduled task **LNMA Music Video Resume** also resumes an unfinished job at user sign-in. A clean stop marker suppresses that automatic resume.

## Production files

- `production/timeline.json` stores decoded sample count, beat times, sections, energy and vocal estimates at half-second intervals.
- `production/identity.json` defines the cast, camera treatment, stage geography and reference seeds.
- `production/shots.json` provides frame intervals, musical cues, prompts, angles, actions and diffusion seeds for the entire song.
- `models.lock.json` identifies the visual model files by upstream revision and SHA256. The native engine and auxiliary analysis model are pinned in `bootstrap.py`.
- `analyse.py`, `plan.py`, `render.py` and `finish.py` implement the production stages. To rerun assembly and decoded validation from finished local shots, run `.venv\Scripts\python.exe finish.py`.

The final checked master is written to `output/LNMA_Live_Rap_Metal_Official_Music_Video.mp4`. The assembly stage creates per-shot contact sheets and `output/validation.json`. A review MP4 is renamed to the final filename only after it passes validation. Source audio, reference pictures, model weights, logs, generated clips and masters are local assets excluded from Git.

## Master format

The picture has 1920 × 1080 H.264 frames on a 24 fps grid. The last frame ends at the final audio sample. The stereo soundtrack uses lossless ALAC from the original MP3 decode. Validation checks both track end times, every video timestamp, decoded frame count, darkness, duplicate thumbnails, decoded sample count and a 24-bit PCM hash of the soundtrack. Subject motion and identity still need visual judgment.


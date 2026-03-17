# ZuviScribe

Turn any video on the internet into a cinematic, documentary-style narrated film with AI-generated video, narration, background music, and LaTeX transcripts.

## Architecture

```
Video URL
    │
    ▼
┌──────────┐  ┌────────────┐  ┌───────────────────┐  ┌──────────────────┐
│ Downloader│→ │ Transcriber│→ │ Concept Extractor  │→ │ Script Writer    │
│ (yt-dlp)  │  │ (Whisper)  │  │ (LLaVA 7B)        │  │ (Gemma-2-27B)   │
└──────────┘  └────────────┘  └───────────────────┘  └──────────────────┘
                                                              │
                                                              ▼
┌────────────┐  ┌──────────────────┐  ┌─────────────┐  ┌──────────────────┐
│ Assembler  │← │ Wan 2.1 Animator │← │ Music Gen   │← │ Image Renderer   │
│ (FFmpeg)   │  │ (i2v + FFmpeg)   │  │ (MusicGen)  │  │ (SDXL-Turbo)    │
└────────────┘  └──────────────────┘  └─────────────┘  └──────────────────┘
       │                                                       │
       ▼                                                       ▼
  documentary_output.mp4                              TTS Narration (edge-tts)
```

### Pipeline Steps (8 stages)

| # | Step                  | Tool                        | Description |
|---|-----------------------|-----------------------------|-------------|
| 1 | **Concept Extraction**| `tools/concept_extractor.py`| LLaVA 7B analyses key frames for visual concepts |
| 2 | **Script Writing**    | `tools/script_writer.py`    | Gemma-2-27B writes documentary narration + visual prompts |
| 3 | **TTS Narration**     | `tools/narration_tts.py`    | edge-tts (Microsoft) generates speech, sets timing |
| 4 | **Image Rendering**   | `tools/image_renderer.py`   | SDXL-Turbo renders 3 image variants per scene |
| 5 | **Background Music**  | `tools/music_generator.py`  | MusicGen-small generates mood-matched background music |
| 6 | **Video Animation**   | `tools/wan_animator.py`     | Wan 2.1 image-to-video generates real motion clips |
| 7 | **Assembly**          | `tools/documentary_assembler.py` | FFmpeg composites clips + narration + music |
| 8 | **Summary**           | `run_documentary.py`        | Outputs storyboard.txt, storyboard.json, final video |

### Modes

- **`--animate`**: Full documentary pipeline (default)
- **`--transcribe`**: Download, transcribe, and export a plain text transcript

## Installation

```bash
# Clone and set up
cd video-animator-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# External tools (macOS)
brew install ffmpeg

# Ollama models
ollama pull llava:7b      # Concept extraction
ollama pull gemma2:27b    # Script writing (15 GB)
```

## Usage

### CLI

```bash
# Generate a documentary (downloads video, then runs full pipeline)
python main.py https://www.youtube.com/watch?v=EXAMPLE --animate

# Just transcribe to text
python main.py https://www.youtube.com/watch?v=EXAMPLE --transcribe

# Default (no flag) runs --animate
python main.py https://www.youtube.com/watch?v=EXAMPLE
```

### Python API

```python
from main import video_animate, video_transcribe

# Full documentary pipeline
result = video_animate("https://www.youtube.com/watch?v=EXAMPLE")

# Transcript only
result = video_transcribe("https://www.youtube.com/watch?v=EXAMPLE")
```

## Outputs

```
output/
├── documentary_output.mp4   # Final video
├── storyboard.txt           # Human-readable storyboard
├── storyboard.json          # Machine-readable storyboard
├── transcript.txt           # Plain text transcript (--transcribe)
├── narration/               # Per-scene TTS audio files
├── render_images/           # SDXL-generated images (3 per scene)
├── animate/                 # Video clips per scene
├── background_music.wav     # AI-generated background music
└── download/                # Original video, audio, frames
```

## Configuration

All settings live in `config.yaml`. Edit it to change models, voices, image settings, and more.

Key options:
- `storyboard.model`: Ollama model for script writing (default: `gemma2:27b`)
- `narration.voice`: edge-tts voice (default: `en-US-ChristopherNeural`)
- `image_renderer.variants_per_scene`: Images per scene (default: 3)
- `music.enabled`: Enable/disable AI background music
- `music.volume`: Background music volume (default: 0.12)

Full section reference:

| Section              | What it controls                                  |
|----------------------|---------------------------------------------------|
| `downloader`         | Video format, frame extraction interval & cap     |
| `transcriber`        | Whisper model size, language, device               |
| `concept_extractor`  | Ollama vision model, max concepts per frame        |
| `storyboard`         | LLM model, scene count range, default style        |
| `image_renderer`     | SDXL model, resolution, diffusion steps, CFG       |
| `animation`          | Method (animatediff / img2vid), FPS, motion        |
| `audio_synth`        | Engine (coqui / original), TTS model               |
| `video_assembly`     | Blender toggle, transitions, codec, CRF            |
| `uploader`           | Platform, credentials path, privacy level          |

## Testing

```bash
# All tests
pytest

# With coverage
pytest --cov=tools --cov=agents --cov-report=term-missing

# Single module
pytest tests/test_downloader.py -v
```

All external services (yt-dlp, Whisper, Ollama, FFmpeg, video API) are mocked in tests.

## CI

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push / PR to `main`:

1. **Lint** – `ruff check .`
2. **Test** – `pytest`
3. **Docker** – builds the Docker image to verify the Dockerfile

## Future Extensions

- **Automatic style transfer** – detect the original video's dominant style and generate a contrasting style prompt.
- **Subtitle generation** – multi-language subtitles via Whisper + translation.
- **FastAPI preview UI** – upload a video link and watch the animated preview in-browser.
- **Batch processing** – animate multiple videos from a playlist.

## License

MIT

# Image → Text → Image

A command-line app that reads an image, has a vision model describe it in detail, prints the description, and then gives **only that text** to an image-generation model. The new image is saved locally, with the description and a side-by-side comparison.

```
image → vision model → detailed text description → image-generation model → new image
```

This is **not** image-to-image: the generator never sees the original. The result shows how much of a picture survives being turned into words.

## Quick start (Windows, VS Code + PowerShell)

```
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY="your-gemini-key"
python img2txt2img.py my_photo.jpg
```

The default setup is free: Gemini describes the image (free API key) and Pollinations generates the new one (free, no key).

Output in `output/`:

| File | Content |
|---|---|
| `<name>_<time>_description.txt` | the description from the vision model |
| `<name>_<time>_generated.png` | the image generated from the description |
| `<name>_<time>_comparison.jpg` | original and generated image side by side |

## Options

```
python img2txt2img.py photo.jpg --image-provider huggingface      # FLUX.1-schnell, free HF token
python img2txt2img.py photo.jpg --vision-provider openai --image-provider openai
python img2txt2img.py photo.jpg --vision-provider local --vision-model qwen3.6-35b-a3b
python img2txt2img.py photo.jpg --words 300 --size 1024x1024 --seed 42
python img2txt2img.py photo.jpg --describe-only
```

| Option | Description |
|---|---|
| `--vision-provider` | `gemini` (default), `openai` (gpt-5.6-luna), `local` (LM Studio/Ollama, e.g. Qwen3.6) |
| `--vision-model` | Override the vision model name |
| `--image-provider` | `pollinations` (default, free), `huggingface` (free token), `gemini` (Nano Banana, needs billing), `openai` (gpt-image-2, paid) |
| `--image-model` | Override the image model name |
| `--words` | Target length of the description (40–400, default 180) |
| `--size` | Output size such as `1024x768` (default: keep the original's aspect ratio) |
| `--seed` | Fixed seed for repeatable images (Pollinations, Hugging Face) |
| `-o`, `--output` | Output folder |
| `--describe-only` | Stop after the description |

### Keys and servers

| Provider | Environment variable |
|---|---|
| Gemini | `GEMINI_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Hugging Face | `HF_TOKEN` (free at huggingface.co/settings/tokens) |
| Pollinations | none (optional `POLLINATIONS_TOKEN` for higher limits) |
| Local vision server | `LOCAL_BASE_URL` (default `http://localhost:1234/v1`, LM Studio) |

## How it works

1. **Read and check the image** (Pillow): the file type and size are checked, the file must really be an image, and large images are scaled down to at most 1536 px before upload, which saves tokens.
2. **Describe** (`providers.describe_image`): the image is sent as a base64 data URL through the OpenAI-compatible Chat Completions API. The prompt asks for one detailed paragraph covering the subject, setting, composition and camera, lighting, colours, style and mood. That is the information an image generator needs. Text inside the image is treated as part of the picture, never as an instruction.
3. **Print** the description and save it as a `.txt` file.
4. **Generate** (`providers.generate_image`): the description is the only input to the image model. The original's aspect ratio is kept by default.
5. **Save** the generated image and a side-by-side comparison.

All providers sit behind two functions (`describe_image`, `generate_image`), so switching models doesn't change the pipeline.

## Files

```
img2txt2img.py     command-line app and pipeline
providers.py       vision and image-generation providers
requirements.txt
output/            results
```

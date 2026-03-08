Create a virsual environment
```bash
uv venv
```
Activate with
```bash
overlay use .venv/bin/activate.nu
```

To toggle between CPU and CUDA, use
```bash
uv sync --extra cpu
```
and
```bash
uv sync --extra cu130
```
respectively

# GPT-2 implementation
To use OpenAI's OG GPT-2 locally, run this command:
```bash
uvx hf download gpt2 --local-dir gpt2
```
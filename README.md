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
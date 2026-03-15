# Manual HuggingFace Model Download Guide

If automatic downloads fail, you can download models manually using these methods.

## Option 1: Using `huggingface-cli` (Easiest)

### Install HuggingFace CLI
```bash
pip install huggingface-hub
```

### Download a Model
```bash
# Create the folder first
mkdir -p ~/HuggingFaceModels

# Download Mistral 7B (Recommended - most compatible)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 --cache-dir ~/HuggingFaceModels
```

### Other Models
```bash
# Llama 2 7B Chat
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --cache-dir ~/HuggingFaceModels

# Mistral 7B Base
huggingface-cli download mistralai/Mistral-7B-v0.1 --cache-dir ~/HuggingFaceModels

# MiniMax M2.1 (requires transformers>=4.36)
huggingface-cli download MiniMaxAI/MiniMax-M2.1 --cache-dir ~/HuggingFaceModels
```

### With HuggingFace Token (for private models)
```bash
huggingface-cli login
# Then paste your token from https://huggingface.co/settings/tokens

huggingface-cli download MiniMaxAI/MiniMax-M2.1 --cache-dir ~/HuggingFaceModels
```

---

## Option 2: Using Python Script

### Simple Download Script
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Setup
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model_folder = Path.home() / "HuggingFaceModels" / model_id.replace("/", "-")
model_folder.mkdir(parents=True, exist_ok=True)

# Download tokenizer
print(f"Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=str(model_folder),
    trust_remote_code=True
)
print(f"✓ Tokenizer saved to {model_folder}")

# Download model
print(f"Downloading model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=str(model_folder),
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cpu"
)
print(f"✓ Model saved to {model_folder}")
```

---

## Option 3: Manual Download from Web

### Using Browser
1. Go to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
2. Click "Files and versions"
3. Download individual files to `~/HuggingFaceModels/mistralai-Mistral-7B-Instruct-v0.1/`

⚠️ **Note**: This is slow and requires downloading many files. Use Option 1 or 2 instead.

---

## Recommended Models (by compatibility)

### ✅ Best for Quick Testing (Smallest, Fastest)
```
Model: mistralai/Mistral-7B-Instruct-v0.1
Size: ~14 GB
Memory: ~16 GB RAM
Download time: ~30 minutes (on fiber)
Status: Fully compatible ✓
```

### ✅ Good for Chat (Llama-based)
```
Model: meta-llama/Llama-2-7b-chat-hf
Size: ~13.5 GB
Memory: ~16 GB RAM
Download time: ~30 minutes
Status: Fully compatible ✓
Requires: Accept license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

### ⚠️ Advanced (Requires Setup)
```
Model: MiniMaxAI/MiniMax-M2.1
Size: 450 GB (full) / 60 GB (quantized)
Memory: 500+ GB for full / 64 GB+ for quantized
Download time: Several hours
Status: Requires transformers>=4.36
Installation:
  pip install --upgrade transformers torch
  pip install accelerate
```

---

## Troubleshooting Download Issues

### Issue: "Connection timeout"
**Solution**:
```bash
# Retry with timeout and chunk size
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 \
  --cache-dir ~/HuggingFaceModels \
  --resume-download
```

### Issue: "Out of disk space"
**Solution**:
- Mistral 7B needs ~14 GB
- Llama 2 7B needs ~13.5 GB
- MiniMax M2.1 needs ~450 GB (or ~60 GB with quantization)

Check available space:
```bash
df -h ~/HuggingFaceModels
```

### Issue: "Module not found" when loading model
**Solution**:
```bash
# Upgrade transformers
pip install --upgrade transformers torch

# For MiniMax specifically
pip install transformers>=4.36
```

### Issue: "Permission denied" when writing to folder
**Solution**:
```bash
# Make sure you own the folder
sudo chown -R $USER ~/HuggingFaceModels
chmod -R u+w ~/HuggingFaceModels
```

---

## After Downloading

Once you have models in `~/HuggingFaceModels/`, the Android Agent will:
1. **Detect them automatically** (shows ✓ downloaded)
2. **Skip re-downloading**
3. **Use them immediately**

Example folder structure after download:
```
~/HuggingFaceModels/
├── mistralai-Mistral-7B-Instruct-v0.1/
│   ├── config.json
│   ├── tokenizer.json
│   ├── model.safetensors
│   └── ...
├── meta-llama-Llama-2-7b-chat-hf/
│   ├── config.json
│   ├── tokenizer.json
│   ├── model.safetensors
│   └── ...
└── ...
```

---

## System Requirements

### For Mistral 7B (Recommended)
- **RAM**: 16 GB minimum (32 GB recommended)
- **Disk**: 20 GB free
- **Download time**: 30-60 minutes

### For Llama 2 7B
- **RAM**: 16 GB minimum (32 GB recommended)
- **Disk**: 20 GB free
- **Download time**: 30-60 minutes

### For MiniMax M2.1 (Full)
- **RAM**: 512 GB or more
- **Disk**: 500 GB free
- **Download time**: Several hours
- **Better approach**: Use quantized version (~60 GB)

---

## Commands Cheat Sheet

```bash
# Create folder
mkdir -p ~/HuggingFaceModels

# Download Mistral 7B (recommended)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 --cache-dir ~/HuggingFaceModels

# Download Llama 2 7B Chat
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --cache-dir ~/HuggingFaceModels

# List downloaded models
ls -lh ~/HuggingFaceModels/

# Check folder size
du -sh ~/HuggingFaceModels/
```

---

## Need Help?

- **HuggingFace Hub Issues**: https://github.com/huggingface/huggingface_hub/issues
- **Transformers Library**: https://github.com/huggingface/transformers/issues
- **Model Cards**: https://huggingface.co/models


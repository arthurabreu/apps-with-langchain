# Quick Start: Download & Use HuggingFace Models

## 🚀 Fastest Path (5 minutes to ready)

### Step 1: Install CLI Tool (one-time setup)
```bash
pip install huggingface-hub
```

### Step 2: Download Mistral (Recommended)
```bash
mkdir -p ~/HuggingFaceModels
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 --cache-dir ~/HuggingFaceModels
```
**⏱️ Takes 30-60 minutes on decent internet**

### Step 3: Use in Android Agent
```bash
python src/main.py
→ Select: 1 (Android Code-Gen Agent)
→ Select: 2 (Local HuggingFace Model)
→ Select: 1 (Mistral-7B-Instruct)
→ See: "✓ (downloaded)"
→ Ready to use!
```

---

## 📊 Model Comparison

| Model | Size | Speed | Quality | Status |
|-------|------|-------|---------|--------|
| **Mistral 7B Instruct** ⭐ | 14 GB | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Good | ✅ Stable |
| Llama 2 7B Chat | 13.5 GB | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Good | ✅ Stable |
| MiniMax M2.1 | 450 GB | 🐌 Slow | ⭐⭐⭐⭐⭐ Best | ⚠️ Complex |
| DeepSeek V3.2 | 690 GB | 🐢 Very Slow | ⭐⭐⭐⭐⭐ Excellent | ⚠️ Massive |

---

## 📍 File Location

All downloaded models go to:
```
~/HuggingFaceModels/
├── mistralai-Mistral-7B-Instruct-v0.1/
├── meta-llama-Llama-2-7b-chat-hf/
└── ...
```

**Easy access**: Open Files → Home → HuggingFaceModels

---

## 💡 Common Issues & Fixes

### "Model not found" Error
```bash
# Retry with resume
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 \
  --cache-dir ~/HuggingFaceModels \
  --resume-download
```

### Download Too Slow
```bash
# Set HuggingFace token (increases speed)
huggingface-cli login
# Paste token from https://huggingface.co/settings/tokens
```

### "Out of disk space"
- Mistral: needs 20 GB free
- Llama 2: needs 20 GB free
- MiniMax: needs 500 GB free!

Check space:
```bash
df -h ~
```

### "Out of memory" During Download
- Close other applications
- Use smaller model (Mistral instead of MiniMax)

---

## ⚡ Use Without Downloading First

Don't want to pre-download? No problem!

```bash
python src/main.py
→ Android Agent
→ Local HuggingFace Model
→ Select model (1, 2, or 3)
→ Answer "y" to download prompt
→ Wait while it downloads
→ Use immediately after
```

---

## 🎯 Recommended Setup

```bash
# Create folder
mkdir -p ~/HuggingFaceModels

# Download Mistral (one-time, ~30 mins)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 --cache-dir ~/HuggingFaceModels

# Done! Now use Android Agent anytime
python src/main.py
```

---

## 📚 Detailed Guides

- **Manual downloads**: See `DOWNLOAD_MODELS.md`
- **Troubleshooting**: See `FIX_SUMMARY.md`
- **Requirements**: See `FIX_SUMMARY.md` → System Requirements

---

## 🔧 Command Reference

```bash
# Check if HF tools installed
huggingface-cli --version

# Download a model
huggingface-cli download {MODEL_ID} --cache-dir ~/HuggingFaceModels

# List downloaded models
ls -lh ~/HuggingFaceModels/

# Check folder size
du -sh ~/HuggingFaceModels/

# Login to HuggingFace (for private models)
huggingface-cli login

# Logout
huggingface-cli logout
```

---

## ✅ Verification

After downloading, check:

```bash
# Model should exist
ls ~/HuggingFaceModels/mistralai-Mistral-7B-Instruct-v0.1/

# Should see files like:
# - config.json
# - tokenizer.json
# - model.safetensors
```

---

## 🎓 Learning Resources

- HuggingFace Models: https://huggingface.co/models
- Transformers Docs: https://huggingface.co/docs/transformers
- Model Cards: Click any model on huggingface.co

---

## 💬 Have Questions?

1. **"Which model should I use?"** → Mistral 7B Instruct (option 1)
2. **"How long does download take?"** → 30-60 mins on good internet
3. **"Can I delete a model?"** → Yes: `rm -rf ~/HuggingFaceModels/model-name`
4. **"Will it re-download?"** → No, it skips if already there ✓
5. **"Need private model?"** → Run `huggingface-cli login` first

---

**Ready? Run this:**
```bash
mkdir -p ~/HuggingFaceModels
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 --cache-dir ~/HuggingFaceModels
```

Then open Android Agent! 🚀


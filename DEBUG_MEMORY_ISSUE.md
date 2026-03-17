# Debug Checklist: Memory Spike to 98%

## What Was Wrong

The original code used `HuggingFacePipeline.from_model_id()` which **doesn't pass quantization_config and max_memory parameters** to the underlying model loader. This meant:
- ✗ 4-bit quantization was **ignored**
- ✗ Memory limits were **ignored**
- ✗ Models loaded at full float32 precision (~14-28 GB)
- ✗ System went to 98% memory

## What Was Fixed

Updated `src/android_agent/provider_factory.py` to:
1. Load model directly with `AutoModelForCausalLM.from_pretrained()`
2. **Explicitly pass** `quantization_config` (4-bit NF4)
3. **Explicitly pass** `max_memory` dict (85% limits)
4. **Explicitly pass** `device_map="auto"` (GPU first)
5. Wrap the loaded model with HuggingFacePipeline

Now the parameters are **actually enforced**.

---

## Step-by-Step Verification

### 1. **Check bitsandbytes is installed**
```bash
python3 -c "import bitsandbytes; print('✓ bitsandbytes installed')"
```
If fails: `pip install bitsandbytes`

### 2. **Check CUDA is available**
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```
Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070 Ti
```

### 3. **Check quantization config loads**
```bash
python3 -c "
from transformers import BitsAndBytesConfig
import torch
cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
print('✓ BitsAndBytesConfig created successfully')
print(f'Config: {cfg}')
"
```

### 4. **Test with a small model first**
Don't start with a 7B model. Use Mistral-3B or TinyLlama first:

```bash
python3 src/main.py
# Select: Test Local HuggingFace model
# Choose: Mistral-7B-Instruct (smaller, tests quantization)
```

### 5. **Monitor with nvidia-smi in another terminal**
```bash
nvidia-smi -l 1  # Updates every 1 second
```

Watch for:
- ✓ **GPU-Util:** Should rise to 50-100% (not 0%)
- ✓ **Memory-Usage:** Should be 3-5 GB (not 14-28 GB)
- ✓ **GPU Processes:** Should show your Python process

Example output during model loading:
```
GPU  Name                Persistence-M | Bus-Id  Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX... Off | 00000000:01:00.0 On      |                  N/A |
| 0%   45C    P0              80W /  285W |   5500 MiB / 12282 MiB |     85%      Default |
```

### 6. **Check memory doesn't spike on system**
While GPU is loading, check system memory with:
```bash
free -h  # Show once per second
watch -n 1 free -h
```

Expected:
- ✓ System RAM stays **below 26 GB** (85% cap)
- ✓ No swap usage or minimal
- ✓ System stays responsive

---

## If Memory Still Goes to 98%

### Diagnosis

**Check 1: Is quantization actually loading?**
```bash
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "10000MiB", "cpu": "26000MiB"},
    token="YOUR_HF_TOKEN"  # or set HUGGINGFACE_API_KEY
)

print(f"Model dtype: {model.dtype}")  # Should be torch.uint8 (quantized)
print(f"Model device: {model.device}")  # Should be cuda:0
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
EOF
```

**Check 2: Is VS Code using GPU memory?**
```bash
nvidia-smi | grep -i "code\|vscode"
```
If VS Code is taking GPU memory, close it before loading models.

**Check 3: Are other GPU processes hogging memory?**
```bash
nvidia-smi  # Look at GPU Memory-Usage
```
Kill other GPU apps:
```bash
killall java  # If Java is hogging GPU
pkill -f "process_name"
```

### Solutions

**If quantization still not working:**
1. Reinstall bitsandbytes: `pip install --upgrade bitsandbytes`
2. Check CUDA toolkit: `nvcc --version` (should match driver)
3. Try pre-built wheel: https://github.com/TimDettmers/bitsandbytes/releases

**If memory cap not working:**
1. Verify max_memory format: `{0: "10000MiB", "cpu": "26000MiB"}`
2. Check available memory: `nvidia-smi` and `free -h`
3. Reduce max_memory further if needed (e.g., 8GB instead of 10GB)

**If GPU not being used:**
1. Check: `torch.cuda.is_available()` returns `True`
2. Check: `torch.cuda.get_device_name(0)` returns your GPU name
3. Check: No CUDA errors in output

---

## Performance Expectations (After Fix)

| Step | VRAM Used | System RAM | CPU % | Status |
|------|-----------|-----------|-------|--------|
| Load model | 1-2 GB | 1-2 GB | 20-30% | Loading shards |
| After load | 4-5 GB | 2-3 GB | 5-10% | Idle |
| During inference | 5-6 GB | 2-4 GB | 50-70% | Active |
| **Never** | **>10.5 GB** | **>26 GB** | **>85%** | ✓ Capped |

---

## Files Changed in Fix

- `src/android_agent/provider_factory.py` — Now loads model directly with quantization params

---

## Next Steps

1. Run verification checks above
2. Try loading Mistral-7B-Instruct
3. Monitor `nvidia-smi -l 1` and `free -h`
4. If working: try larger models
5. If still broken: send output of checks above for debugging

---

*Last updated: 2026-03-17*

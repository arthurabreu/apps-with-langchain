# User Interface Package 🖥️

> **For Android Engineers**: Think of this as your **presentation layer** - like Activities, Fragments, and ViewModels in Android. This package handles all user interactions, displays information, and manages the application flow.

## What This Package Does

This package is responsible for managing all user interface and interaction logic in our LangChain application. Just like how in Android you have Activities that handle user input and display information, this package handles:

- **Main Application Entry Point** (like MainActivity in Android)
- **Console Menu System** (like navigation between fragments)
- **User Input/Output Management** (like handling button clicks and displaying data)
- **Application Flow Control** (like navigation logic)

## Files in This Package

### 1. `main.py` - The Main Application Controller 🎮

**What it does**: This is like your MainActivity - it's the main entry point that coordinates everything and provides a menu-driven interface for testing different AI models.

**Key Components**:

#### Application Structure - Line by Line

**Main Menu System**:
```python
def main():
    print("LangChain Model Testing Lab")
    print("=" * 60)
    print("Goal: Test and compare different LLM models (Local, OpenAI, Claude)")
    
    print_api_key_status()  # Show which API keys are configured
    
    while True:  # Main application loop (like Android's activity lifecycle)
        print("\nMAIN MENU")
        print("1. Test Local Hugging Face model")
        print("2. Test OpenAI model") 
        print("3. Test Claude model")
        print("4. Compare all available models")
        print("5. Learn about LangChain concepts")
        print("6. View example code snippets")
        print("7. Check memory usage")
        print("8. Clean up model memory")
        print("9. View system information")
        print("10. Exit")
        
        choice = input("\nEnter your choice (1-10): ").strip()
        
        # Handle user selection (like onOptionsItemSelected in Android)
        if choice == "1":
            test_local_model()
        elif choice == "2":
            test_openai_model()
        # ... etc
```

**Android Equivalent**: This is like having a MainActivity with a navigation drawer or bottom navigation that lets users choose different features.

#### API Key Status Checker
```python
def print_api_key_status():
    keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Hugging Face": os.getenv("HUGGINGFACE_API_KEY"), 
        "Anthropic": os.getenv("ANTHROPIC_API_KEY")
    }
    
    print("\nAPI Key Status:")
    print("-" * 40)
    for name, key in keys.items():
        is_ok = key and "your-" not in key.lower() and key.strip() != ""
        print(f"{name:13}: {'[OK] Configured' if is_ok else '[X] Missing'}")
```

- **Lines 1-5**: Create a dictionary of all API keys we need to check
- **Lines 7-10**: Display a formatted status report
- **Line 11**: Check if each key is properly configured (not empty or placeholder)
- **Line 12**: Display status with color-coded indicators
- **Android equivalent**: Like checking permissions or network connectivity status in your app

#### Memory Management System
```python
def check_memory_usage():
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())  # Get current process
        mem_info = process.memory_info()       # Get memory information
        
        print("Memory Usage Report")
        print(f"RSS (Resident Set Size): {mem_info.rss / 1024 / 1024:.2f} MB")
        print(f"VMS (Virtual Memory Size): {mem_info.vms / 1024 / 1024:.2f} MB")
        print(f"Percent of system RAM: {process.memory_percent():.1f}%")
        
        # Get system-wide memory info
        system_memory = psutil.virtual_memory()
        print(f"System Total: {system_memory.total / 1024 / 1024:.0f} MB")
        print(f"System Available: {system_memory.available / 1024 / 1024:.0f} MB")
        
        # Check GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    alloc = torch.cuda.memory_allocated(i) / 1024 / 1024
                    cached = torch.cuda.memory_reserved(i) / 1024 / 1024
                    print(f"GPU {i}: Allocated: {alloc:.2f} MB, Cached: {cached:.2f} MB")
        except ImportError:
            print("[INFO] PyTorch not available for GPU memory check")
        
        # Warn if memory usage is high
        if process.memory_percent() > 30:
            print("⚠️  WARNING: High memory usage detected!")
            print("Consider cleaning up model resources.")
            
    except ImportError:
        print("[INFO] psutil not installed. Install with: pip install psutil")
```

**Android Equivalent**: Like monitoring app memory usage with `ActivityManager.getMemoryInfo()` or using profiling tools to track memory leaks.

#### Memory Cleanup System
```python
def cleanup_model_memory():
    print("Cleaning Up Model Memory")
    
    try:
        import gc
        import torch
        
        # Get initial memory usage
        initial_memory = None
        try:
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
        except:
            pass
        
        print("[STEP 1] Running Python garbage collection...")
        gc.collect()  # Force garbage collection
        print("[SUCCESS] Garbage collection completed")
        
        print("[STEP 2] Clearing PyTorch CUDA cache (if using GPU)...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    # Clear GPU memory cache
            torch.cuda.ipc_collect()    # Clean up inter-process communication
            print("[SUCCESS] CUDA cache cleared")
        else:
            print("[INFO] No GPU detected, skipping CUDA cache clear")
        
        print("[STEP 3] Clearing other caches...")
        # Clear any remaining model references
        if 'local_model' in globals():
            try:
                local_model.cleanup()
                del local_model
                print("[SUCCESS] Local model cleaned up")
            except:
                pass
        
        # Force another garbage collection
        gc.collect()
        
        # Report memory freed
        if initial_memory:
            try:
                final_memory = process.memory_info().rss
                freed = (initial_memory - final_memory) / 1024 / 1024
                if freed > 0:
                    print(f"[SUCCESS] Freed approximately {freed:.2f} MB of memory")
                else:
                    print("[INFO] Memory usage remained about the same")
            except:
                pass
                
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
```

**Android Equivalent**: Like calling `System.gc()`, clearing image caches, or releasing resources in `onDestroy()` methods.

#### Model Testing Functions

**Local Model Testing**:
```python
def test_local_model():
    print("[TEST] Local Hugging Face Model Demo")
    
    try:
        from core.langchain_huggingface_local import LocalHuggingFaceModel
        
        # Check memory before loading
        print("[INFO] Checking memory before loading model...")
        check_memory_usage()
        
        # Initialize local model
        print("[INFO] Initializing local model...")
        local_model = LocalHuggingFaceModel()
        
        # Test with different prompts
        test_prompts = [
            {
                "name": "Kotlin Coroutine",
                "prompt": "Write a Kotlin function that uses coroutines to fetch data from two APIs concurrently."
            },
            {
                "name": "Kotlin Palindrome", 
                "prompt": "Write a Kotlin function that checks if a string is a palindrome."
            },
            {
                "name": "Explain Coroutines",
                "prompt": "Explain Kotlin coroutines to a beginner. Keep it under 150 words."
            }
        ]
        
        for i, test in enumerate(test_prompts, 1):
            print(f"[{i}/{len(test_prompts)}] Testing: {test['name']}")
            print(f"Prompt: {test['prompt']}")
            
            try:
                response = local_model.generate(test['prompt'])
                print(f"Response: {response}")
                
                # Ask user if they want to continue
                if not prompt_continue_or_skip():
                    break
                    
            except Exception as e:
                print(f"[ERROR] Generation failed: {e}")
                
    except Exception as e:
        print(f"[ERROR] Local model test failed: {e}")
```

**Android Equivalent**: Like testing different features of your app with various inputs and displaying results to the user.

#### System Information Display
```python
def show_system_info():
    import platform
    
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Memory:")
        print(f"  Total: {memory.total / 1024 / 1024 / 1024:.1f} GB")
        print(f"  Available: {memory.available / 1024 / 1024 / 1024:.1f} GB")
        print(f"  Used: {memory.percent}%")
        
        # Provide recommendations
        print(f"Requirements for local models:")
        print(f"  Minimum: 8GB RAM (for small models)")
        print(f"  Recommended: 16GB+ RAM (for 4B+ parameter models)")
        print(f"  GPU: Optional but recommended for speed")
        
        if memory.total < 8 * 1024 * 1024 * 1024:  # Less than 8GB
            print("⚠️  WARNING: System has less than 8GB RAM.")
            print("   Local models may not work well.")
            print("   Consider using cloud models (OpenAI/Claude) instead.")
    except:
        pass
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        print(f"Disk Space:")
        print(f"  Total: {total / 1024 / 1024 / 1024:.1f} GB")
        print(f"  Free: {free / 1024 / 1024 / 1024:.1f} GB")
        
        if free < 10 * 1024 * 1024 * 1024:  # Less than 10GB free
            print("⚠️  WARNING: Less than 10GB disk space free.")
            print("   Model downloads may fail or fill your disk.")
    except:
        pass
```

**Android Equivalent**: Like displaying device information, available storage, and system requirements in your app's settings or about screen.

### 2. `test_claude.py` - Claude Model Test Script 🧪

**What it does**: A focused test script for testing Claude model integration.

**Key Features**:
- Tests Claude model initialization
- Validates API key configuration
- Demonstrates basic usage patterns
- Provides debugging information

**Android Equivalent**: Like having unit tests or integration tests for specific features of your app.

## How to Use This Package

### Running the Main Application
```python
# From the project root directory
python src/userInterface/main.py

# Or if you have the package installed
from userInterface import main
main()
```

### Using Individual Functions
```python
from userInterface import print_api_key_status, check_memory_usage

# Check API key status
print_api_key_status()

# Monitor memory usage
check_memory_usage()
```

### Testing Specific Models
```python
# Test local models
python -c "from userInterface.main import test_local_model; test_local_model()"

# Test OpenAI models  
python -c "from userInterface.main import test_openai_model; test_openai_model()"
```

## Why This Matters for Android Developers

1. **User Experience**: Just like Android UX, this provides clear feedback and status information
2. **Resource Management**: Similar to Android memory management and lifecycle awareness
3. **Error Handling**: Proper error messages and graceful degradation
4. **Navigation Flow**: Menu-driven navigation similar to Android navigation patterns
5. **System Integration**: Checking system capabilities like Android's capability detection

## Common Patterns You'll Recognize

- **Menu Navigation**: Like Android's navigation drawer or bottom navigation
- **Status Indicators**: Like connection status or permission indicators
- **Resource Cleanup**: Like Android's lifecycle methods (`onDestroy`, `onPause`)
- **Progress Feedback**: Like progress bars and loading indicators
- **Error Handling**: Like try-catch blocks with user-friendly error messages
- **System Checks**: Like checking device capabilities or permissions

## Application Flow

```
Start Application
       ↓
Show API Key Status
       ↓
Display Main Menu
       ↓
User Selects Option
       ↓
Execute Selected Function
       ↓
Show Results/Feedback
       ↓
Ask to Continue or Return to Menu
       ↓
Return to Main Menu (or Exit)
```

## Files Structure
```
userInterface/
├── __init__.py          # Package exports
├── main.py             # Main application controller
├── test_claude.py      # Claude model test script
└── README.md          # This documentation
```

This package is like your Android app's UI layer - it handles all user interactions, displays information clearly, and manages the overall application experience!
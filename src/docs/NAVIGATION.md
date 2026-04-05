# Navigation Guide

**📍 Getting help** | [← Back to Index](index.md) | [Core Docs →](../core/index.md)

> Learn how to navigate the project documentation.

## 📚 How to Use These Docs

Each markdown file has **navigation links** at the top and bottom:

```
**📍 Reading Order:** #5 of 10 core docs | [← Back to Index](index.md) | [Next: utils.md →](utils.md)
```

- **📍 Reading Order** — your position in the recommended path
- **← Back to Index** — jump back to the main index anytime
- **Next →** — go to the next recommended file

---

## 🗺️ Core Learning Path (10 docs, ~90 min)

Follow these links in order:

1. [index.md](index.md) ⭐ **START HERE**
   - Architecture overview
   - Dependency graph
   - Quick Python→Kotlin reference
   
2. [interfaces.md](interfaces.md)
   - Contracts & data classes
   - Protocol vs ABC explained
   
3. [exceptions.md](exceptions.md)
   - Exception hierarchy
   - When each exception is raised
   
4. [dependency_injection.md](dependency_injection.md)
   - Manual DI container
   - Service wiring
   
5. [services.md](services.md)
   - Config, logging, validation
   - Concrete implementations
   
6. [utils.md](utils.md)
   - Convenience functions
   - When to use vs. direct container access
   
7. [models/model_factory.md](models/model_factory.md)
   - Factory pattern
   - Provider registry & extensibility
   
8. [models/claude_model.md](models/claude_model.md)
   - Claude model implementation
   - ABC inheritance
   
9. [strategies/standard_generation.md](strategies/standard_generation.md)
   - Synchronous generation
   - LangChain integration
   
10. [strategies/streaming_generation.md](strategies/streaming_generation.md)
    - Streaming generation
    - Real-time output with generators

---

## 📖 Optional Reference Docs

Explore these as needed for specific topics:

- **[token_utils.md](token_utils.md)** — Token counting, cost estimation, usage tracking
  - Read when: implementing token tracking, understanding pricing
  
- **[langchain_huggingface_local.md](langchain_huggingface_local.md)** — Local model inference
  - Read when: exploring local HuggingFace model support, understanding device detection
  
- **[model_comparison.md](model_comparison.md)** — Model benchmarking utilities
  - Read when: comparing model performance, understanding dynamic imports

---

## 🎯 Tips for Best Learning

1. **Start with index.md** — get the big picture first
2. **Follow the numbered path** — each doc builds on previous ones
3. **Use the navigation links** — each file shows next/previous
4. **Keep index.md open** — refer back to the architecture diagram
5. **Read alongside the source code** — open the `.py` file next to its `.md`
6. **Take notes** — the Python→Kotlin cheat sheets are gold for Android devs

---

## 🔍 Quick Reference

**Find a doc by topic:**

| Topic | File |
|-------|------|
| System architecture | [index.md](index.md) |
| Interfaces & contracts | [interfaces.md](interfaces.md) |
| Error handling | [exceptions.md](exceptions.md) |
| Dependency injection | [dependency_injection.md](dependency_injection.md) |
| Services (config, logging) | [services.md](services.md) |
| Creating models | [models/model_factory.md](models/model_factory.md) |
| Claude implementation | [models/claude_model.md](models/claude_model.md) |
| Text generation (sync) | [strategies/standard_generation.md](strategies/standard_generation.md) |
| Text generation (streaming) | [strategies/streaming_generation.md](strategies/streaming_generation.md) |
| Token tracking & costs | [token_utils.md](token_utils.md) |
| Local model support | [langchain_huggingface_local.md](langchain_huggingface_local.md) |
| Model comparison | [model_comparison.md](model_comparison.md) |

---

## ⚡ Quick Start

1. Open [index.md](index.md)
2. Read the architecture overview (5 min)
3. Click "Next →" to go to [interfaces.md](interfaces.md)
4. Keep clicking "Next →" to follow the learning path
5. Stop when you have enough understanding (or continue to the end)

**Estimated time:** 90 minutes for complete core path, 30-45 min for key concepts

---

**[← Previous](QUICK_START.md) | [Back to Index](index.md) | [Next →](../core/index.md)**

*Ready to dive into the code? Start with the Core Architecture guide*

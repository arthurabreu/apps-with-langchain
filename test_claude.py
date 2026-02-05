#!/usr/bin/env python3
"""
Test script for ClaudeModel with TokenManager integration
"""

import sys
import os
sys.path.append('src/core')

from claude_model import ClaudeModel

def test_claude_model():
    """Test ClaudeModel initialization and basic functionality"""
    print("Testing ClaudeModel with TokenManager integration...")
    
    try:
        # Test initialization without API key (should show error message)
        print("\n1. Testing initialization without API key...")
        model = ClaudeModel()
        print("   ✗ Should have failed - API key not configured")
    except ValueError as e:
        print(f"   ✓ Expected error: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
    
    # Test with mock API key
    print("\n2. Testing initialization with mock API key...")
    os.environ["ANTHROPIC_API_KEY"] = "test-key-for-initialization"
    
    try:
        model = ClaudeModel(model_name="claude-3-haiku-20240307")
        print("   ✓ ClaudeModel initialized successfully")
        
        # Test model info
        info = model.get_model_info()
        print(f"   ✓ Model info: {info}")
        
        # Test token manager initialization
        if hasattr(model, 'token_manager'):
            print("   ✓ TokenManager initialized successfully")
        else:
            print("   ✗ TokenManager not found")
            
    except Exception as e:
        print(f"   ✗ Initialization failed: {e}")
    
    print("\n✓ All tests completed!")

if __name__ == "__main__":
    test_claude_model()
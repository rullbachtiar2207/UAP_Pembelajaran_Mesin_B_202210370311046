#!/usr/bin/env python3
"""
🍎 Mac M4 Pro Setup Verification Script
Verifikasi setup TensorFlow dan Metal acceleration
"""

import platform
import sys

def check_system():
    print("=" * 70)
    print("🍎 Mac M4 Pro Setup Verification")
    print("=" * 70)
    
    # System Info
    print(f"\n📱 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {platform.python_version()}")
    
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        print("   ✅ Apple Silicon detected!")
    else:
        print("   ⚠️  Not running on Apple Silicon")

def check_tensorflow():
    print(f"\n🤖 TensorFlow Setup:")
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow: {tf.__version__}")
        
        # Check devices
        devices = tf.config.list_physical_devices()
        print(f"\n🔧 Available Devices:")
        
        has_gpu = False
        for device in devices:
            print(f"   - {device.device_type}: {device.name}")
            if device.device_type == 'GPU':
                has_gpu = True
        
        if has_gpu:
            print("\n   🚀 Metal GPU Acceleration: ENABLED ✅")
        else:
            print("\n   ⚠️  Metal GPU not detected")
            print("   Install: pip install tensorflow-metal")
            
    except ImportError as e:
        print(f"   ❌ TensorFlow not installed: {e}")
        print("   Install: pip install tensorflow-macos tensorflow-metal")

def check_other_packages():
    print(f"\n📦 Other Packages:")
    
    packages = {
        'transformers': 'Hugging Face Transformers',
        'nltk': 'Natural Language Toolkit',
        'sklearn': 'Scikit-learn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'streamlit': 'Streamlit'
    }
    
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {name}: {version}")
        except ImportError:
            print(f"   ❌ {name}: Not installed")

def check_nltk_data():
    print(f"\n📚 NLTK Data:")
    try:
        import nltk
        
        required_data = ['stopwords', 'punkt', 'wordnet']
        for data in required_data:
            try:
                nltk.data.find(f'corpora/{data}' if data != 'punkt' else f'tokenizers/{data}')
                print(f"   ✅ {data}: Downloaded")
            except LookupError:
                print(f"   ❌ {data}: Not downloaded")
                print(f"      Run: python -m nltk.downloader {data}")
    except ImportError:
        print(f"   ❌ NLTK not installed")

def test_metal_performance():
    print(f"\n⚡ Metal Performance Test:")
    try:
        import tensorflow as tf
        import time
        
        # Simple matrix multiplication test
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            start = time.time()
            c = tf.matmul(a, b)
            _ = c.numpy()  # Force execution
            gpu_time = time.time() - start
            
        print(f"   GPU computation time: {gpu_time*1000:.2f}ms")
        print(f"   ✅ Metal acceleration is working!")
        
    except Exception as e:
        print(f"   ⚠️  Metal test failed: {e}")
        print(f"   GPU may not be available")

def main():
    check_system()
    check_tensorflow()
    check_other_packages()
    check_nltk_data()
    test_metal_performance()
    
    print("\n" + "=" * 70)
    print("✅ Verification Complete!")
    print("=" * 70)
    
    print("\n💡 Next Steps:")
    print("   1. If any packages are missing, install with: pip install -r requirements.txt")
    print("   2. Download NLTK data: python -m nltk.downloader stopwords punkt wordnet")
    print("   3. Run notebooks in order: 01 -> 02 -> 03 -> 04 -> 05")
    print("   4. Launch Streamlit app: streamlit run app.py")
    print()

if __name__ == "__main__":
    main()

"""
Utilities for GPU detection and configuration.
"""

import os
import tensorflow as tf
import subprocess
import platform
from typing import Dict, Any, Optional, List, Tuple

def get_cuda_info() -> Dict[str, Any]:
    """
    Get detailed information about CUDA installation and GPU availability.
    
    Returns:
        Dictionary containing CUDA and GPU information
    """
    info = {
        "tf_gpu_available": False,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_devices": [],
        "nvidia_smi_output": None
    }
    
    # Check if TensorFlow can see any GPUs
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            info["tf_gpu_available"] = True
            info["gpu_devices"] = [str(gpu) for gpu in gpus]
            
            # Get CUDA version from TensorFlow build info
            if hasattr(tf.sysconfig, 'get_build_info'):
                build_info = tf.sysconfig.get_build_info()
                if 'cuda_version' in build_info:
                    info["cuda_version"] = build_info['cuda_version']
                    info["cuda_available"] = True
                if 'cudnn_version' in build_info:
                    info["cudnn_version"] = build_info['cudnn_version']
    except Exception as e:
        print(f"Error checking TensorFlow GPU availability: {e}")
    
    # Try to get CUDA info from environment variables
    if 'CUDA_VERSION' in os.environ:
        info["cuda_version"] = os.environ['CUDA_VERSION']
        info["cuda_available"] = True
    
    # Try running nvidia-smi on supported platforms
    if platform.system() in ['Linux', 'Windows']:
        try:
            nvidia_smi = subprocess.run(
                ['nvidia-smi'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            if nvidia_smi.returncode == 0:
                info["nvidia_smi_output"] = nvidia_smi.stdout
                info["cuda_available"] = True
                # Try to extract CUDA version from nvidia-smi output
                if not info["cuda_version"]:
                    import re
                    cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', nvidia_smi.stdout)
                    if cuda_match:
                        info["cuda_version"] = cuda_match.group(1)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass  # nvidia-smi not available
    
    return info

def print_cuda_info(info: Dict[str, Any]) -> None:
    """
    Print formatted CUDA and GPU information.
    
    Args:
        info: Dictionary containing CUDA and GPU information
    """
    print("\n===== GPU and CUDA Information =====")
    print(f"TensorFlow GPU Available: {info['tf_gpu_available']}")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_version']:
        print(f"CUDA Version: {info['cuda_version']}")
    if info['cudnn_version']:
        print(f"cuDNN Version: {info['cudnn_version']}")
    
    if info['gpu_devices']:
        print(f"\nGPU Devices: {len(info['gpu_devices'])}")
        for i, device in enumerate(info['gpu_devices']):
            print(f"  {i+1}: {device}")
    
    if info['nvidia_smi_output']:
        print("\n----- nvidia-smi Output -----")
        # Print only the first few lines to avoid too much output
        lines = info['nvidia_smi_output'].split('\n')
        print('\n'.join(lines[:20]))
        if len(lines) > 20:
            print("...")
    
    print("==================================\n")

def configure_gpu() -> bool:
    """
    Configure GPU settings for TensorFlow and check CUDA availability.
    
    Returns:
        bool: True if GPU is available and configured, False otherwise
    """
    try:
        # Get CUDA info
        cuda_info = get_cuda_info()
        print_cuda_info(cuda_info)
        
        # Configure GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for GPU: {gpu}")
            
            # Set visible devices
            tf.config.set_visible_devices(gpus, 'GPU')
            
            print(f"GPU successfully configured: Found {len(gpus)} GPU(s)")
            return True
        else:
            print("No GPU found, training will use CPU only")
            return False
    except Exception as e:
        print(f"Error configuring GPU: {e}")
        return False

def force_gpu_usage():
    """
    Attempt to force TensorFlow to recognize and use the GPU by setting various environment variables
    and configurations that might help with specific GPU issues.
    
    Returns:
        bool: True if GPU was successfully configured, False otherwise
    """
    try:
        # Try to set environment variables that might help
        import os
        
        # Common environment variables that help with GPU detection
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs
        
        # For older NVIDIA GPUs
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '1'
        
        # For Windows specifically
        if os.name == 'nt':
            os.environ['CUDA_CACHE_DISABLE'] = '0'
            
        import tensorflow as tf
        
        # Force device placement logging
        tf.debugging.set_log_device_placement(True)
        
        # Try to see if GPU is visible now
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Configure all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Verify GPU is being used
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    c = tf.matmul(a, b)
                    print("Test GPU computation result:", c)
                    
                print("GPU successfully configured and tested")
                return True
            except Exception as e:
                print(f"Error configuring GPU: {e}")
                return False
        else:
            print("No GPUs found after attempted force configuration")
            
            # Last resort - check if CUDA is installed but not being detected
            try:
                import subprocess
                output = subprocess.check_output(['nvidia-smi'], shell=True)
                print("NVIDIA-SMI output:")
                print(output.decode('utf-8'))
                print("CUDA appears to be installed but TensorFlow can't detect the GPU")
                print("This might be a version incompatibility between CUDA and TensorFlow")
            except:
                pass
                
            return False
    except Exception as e:
        print(f"Error trying to force GPU usage: {e}")
        return False

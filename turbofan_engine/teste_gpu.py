import tensorflow as tf

def check_gpu():
    """
    Checks if TensorFlow can detect and use a GPU.
    Prints detailed information about the detected devices.
    """
    print(f"TensorFlow version: {tf.__version__}\n")

    # List all physical GPUs detected
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPU detected by TensorFlow.")
    else:
        print(f"✅ TensorFlow detected {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  [{i}] {gpu.name}")

    # Optional: print logical devices (visible to TF)
    logical_gpus = tf.config.list_logical_devices('GPU')
    if logical_gpus:
        print("\nLogical GPUs available to TensorFlow:")
        for lg in logical_gpus:
            print(f"  - {lg.name}")

    # Check if TensorFlow is actually using GPU for ops
    print("\nRunning a small test operation...")
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        a = tf.random.uniform((1000, 1000))
        b = tf.random.uniform((1000, 1000))
        c = tf.matmul(a, b)
        print(f"Computation device: {c.device}")

if __name__ == "__main__":
    check_gpu()

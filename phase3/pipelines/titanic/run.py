import subprocess
import sys
import os

def run_pipeline():
    # Detect if we are on Windows or Colab
    is_colab = 'google.colab' in sys.modules
    
    print(f"🖥️ Environment: {'Colab' if is_colab else 'Local Windows'}")
    
    try:
        # Step 1: Training
        subprocess.run([sys.executable, "train.py"], check=True)
        
        # Step 2: Evaluation
        subprocess.run([sys.executable, "evaluate.py"], check=True)
        
        print("✅ Pipeline Complete!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed at step: {e.cmd}")

if __name__ == "__main__":
    run_pipeline()
import os
import sys

def setup_environment(subfolder="kaggle-b2"):
    """Handles paths for subfolder-based repos."""
    is_kaggle = os.path.exists('/kaggle/working') # Detect if we are on Kaggle
    
    if is_kaggle:
        # On Kaggle, the repo root is /kaggle/working/research-notebook
        # We need to add the subfolder to the path
        root = f"/kaggle/working/research-notebook/{subfolder}"
        artifacts = "/kaggle/working/artifacts"
    else:
        # Local E: Drive logic
        # Finds the 'kaggle-b2' directory relative to the notebook
        root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
        artifacts = os.path.join(os.getcwd(), "artifacts")
        
    # Ensure the 'kaggle-b2' folder is in Python's search path
    if root not in sys.path:
        sys.path.append(root)
        
    os.makedirs(artifacts, exist_ok=True)
    return root, artifacts, is_kaggle
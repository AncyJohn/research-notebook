from setuptools import setup, find_packages

setup(
    name="shared",
    version="0.1.0",
    packages=find_packages(), # This finds 'shared' because it has __init__.py
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm"
    ], 
)
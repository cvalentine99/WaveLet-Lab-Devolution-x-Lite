"""
GPU RF Forensics Engine - Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="rf-forensics",
    version="1.0.0",
    description="GPU-Accelerated RF Forensics Engine for NVIDIA RTX 4090",
    author="RF Forensics Team",
    packages=["rf_forensics"] + ["rf_forensics." + p for p in find_packages()],
    package_dir={"rf_forensics": "."},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "websockets>=12.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "aiofiles>=23.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-benchmark>=4.0.0",
            "pytest-cov>=4.1.0",
            "ipython>=8.0.0",
            "jupyterlab>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rf-forensics=rf_forensics.pipeline.orchestrator:main",
            "rf-verify=rf_forensics.scripts.verify_environment:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Signal Processing",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

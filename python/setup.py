"""
Setup script for OSS_SRV Python SDK
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory.parent / "README.md").read_text()

setup(
    name="oss_srv",
    version="1.0.0",
    author="Brian Worthington",
    author_email="brian@example.com",
    description="High-performance Python client for OSS_SRV inference server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tcBio/oss_srv",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        # No dependencies for basic client
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "mypy>=0.990",
        ],
        "benchmark": [
            "matplotlib>=3.5.0",
            "numpy>=1.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "oss_srv=oss_srv.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

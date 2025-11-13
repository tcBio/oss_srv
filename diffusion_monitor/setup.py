from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diffusion-monitor",
    version="0.1.0",
    author="Brian Worthington",
    author_email="brian@example.com",
    description="Production-grade monitoring platform for diffusion language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tcBio/oss_srv",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "prometheus-client>=0.19.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.5.0",
        "influxdb-client>=1.40.0",
        "redis>=5.0.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "streamlit>=1.30.0",
        "pandas>=2.1.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "jupyter>=1.0.0",
        ],
        "tensorrt": [
            "tensorrt>=10.0.0",
            "pycuda>=2024.1",
        ],
        "datadog": [
            "datadog>=0.49.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "tensorrt>=10.0.0",
            "datadog>=0.49.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "diffusion-monitor=diffusion_monitor.cli:main",
            "diffusion-exporter=exporters.prometheus_exporter:main",
        ],
    },
)

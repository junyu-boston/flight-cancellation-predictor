"""Setup configuration for Flight Cancellation Predictor package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="flight-cancellation-predictor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready ML system for flight cancellation prediction using AutoGluon TabPFNMix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/flight-cancellation-predictor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
            "pylint>=2.17",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "notebook": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
            "jupyterlab>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "flight-predictor=src.cli.main:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
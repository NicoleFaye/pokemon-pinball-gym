"""Setup script for pokemon-pinball-gym package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pokemon-pinball-gym",
    version="0.1.0",
    author="Nicole Demera",
    author_email="nicolefayedemera@gmail.com",
    description="Pokemon Pinball Gymnasium environment for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NicoleFaye/pokemon-pinball-gym",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
    },
    keywords=["RL", "Pokemon", "Pinball", "Gymnasium", "AI"],
    project_urls={
        "Bug Reports": "https://github.com/NicoleFaye/pokemon-pinball-gym/issues",
        "Source": "https://github.com/NicoleFaye/pokemon-pinball-gym",
        "Documentation": "https://github.com/NicoleFaye/pokemon-pinball-gym#readme",
    },
)
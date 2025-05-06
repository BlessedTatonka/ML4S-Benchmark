from setuptools import setup, find_packages

setup(
    name="ml4s-benchmark",
    version="0.1.0",
    description="A benchmark for evaluating LLMs on SVG editing tasks",
    author="ML4S Team",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "datasets>=2.10.0",
        "openai>=0.27.0",
        "torch>=1.12.0",
        "cairosvg>=2.5.2",
        "pillow>=9.0.0",
    ],
    entry_points={
        "console_scripts": [
            "ml4s=ml4s:run",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchprism",
    version="1.0.1",
    author="Tomasz Szandala",
    author_email="tomasz.szandala@gmail.com",
    description="Principal Image Sections Mapping for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/szandala/TorchPRISM",
    packages=["torchprism"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
        install_requires=[
        "torch >= 1.1",
        "torchvision >= 0.3.0"
    ],
    keywords=["deep-learning", "PCA", "visualization", "interpretability"]
)

# usage: python setup.py sdist bdist_wheel

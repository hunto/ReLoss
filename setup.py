import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reloss",
    version="1.0.0",
    author='ReLoss Contributors',
    description="ReLoss package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['reloss'],
    python_requires='>=3.6',
    url="https://github.com/hunto/ReLoss",
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
    ],
    license='Apache License 2.0',
)

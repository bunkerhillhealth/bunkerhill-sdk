import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="bunkerhill",
  version="0.1.0",
  author="Bunkerhill Health",
  description="SDK for integration with Bunkerhill Health",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/bunkerhillhealth/bunkerhill-sdk",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.9',
  install_requires=[
    'grpcio==1.51.1',
    'grpcio-testing==1.51.1',
    'grpcio-tools==1.51.1',
    'nibabel==5.0.0',
    'numpy>=1.24.0',
  ],
)


import setuptools

with open('README.md', 'r') as fh:
  long_description = fh.read()

setuptools.setup(
  name='bunkerhill-inference-api',
  version='0.0.1',
  author='bunkerhill',
  author_email='engineering@bunkerhillhealth.com',
  description='blib',
  long_description=long_description,
  url='https://github.com/bunkerhillhealth/$URL', # TODO: Add real url
  packages=setuptools.find_packages(),
  classifiers=[
    'Programming Language :: Python :: 3',
    'Operating System :: OS Independent',
  ],
  python_requires='>=3.9',
  install_requires=[
    'asgiref==3.6.0',
    'PyJWT==2.6.0',
    'retry==0.9.2',
    'requests==2.27.1',
  ],
)

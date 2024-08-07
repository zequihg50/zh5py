from setuptools import setup, find_packages

setup(name='zh5',
      version='0.1',
      description='Yet another HDF5 Python reader.',
      url='http://github.com/zequihg50/zh5',
      author='Ezequiel Cimadevilla',
      author_email='ezequiel.cimadevilla@unican.es',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "numcodecs",
          "aiohttp",
      ],
      tests_require=[
          "h5py",
      ],
      zip_safe=False)

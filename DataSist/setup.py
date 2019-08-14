from setuptools import setup

setup(name='Datasist',
      version='0.1',
      description='A Machine learning library that abstracts repetitve functions used by data scientist and machine learning engineers',
      url='https://github.com/risenW/visualz',
      author='Rising Odegua',
      author_email='risingodegua@gmail.com',
      download_url='https://github.com/risenW/visualz/archive/v0.2-alpha.tar.gz', 
      license='MIT',
      install_requires=[
        'pandas',
        'matplotlib',
        'seaborn',
        'numpy',
        'warnings'
    ],
      zip_safe=False
      
      )
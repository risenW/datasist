from setuptools import setup

with open("README.md", "r") as cd:
  long_description = cd.read()


setup(
      name='datasist',
      packages=['datasist'],
      version='1.4',
      license='MIT',
      description='A Machine learning library that abstracts repetitve functions used by data scientist and machine learning engineers',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Rising Odegua',
      author_email='risingodegua@gmail.com',
      url='https://github.com/risenW/datasist',
      keywords=['Data Analysis', 'Feature Engineering', 'Visualization'],
      download_url='https://github.com/risenW/datasist/archive/v1.4.tar.gz', 
      install_requires=[
        'pandas',
        'matplotlib',
        'seaborn',
        'numpy',
        'jupyter',
        'scikit-learn'
        ],
      classifiers=[
        'Development Status :: 5 - Production/Stable',   
        'Intended Audience :: Developers',      
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
      ],
      
      )
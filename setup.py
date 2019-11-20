from setuptools import setup

with open("README.md", "r") as cd:
  long_description = cd.read()


setup(
      name='datasist',
      packages=['datasist'],
      version='0.1',
      license='MIT',
      description='A Machine learning library that abstracts repetitve functions used by data scientist and machine learning engineers',
      long_description=long_description,
      long_description_content_type="text/markdown"
      author='Rising Odegua',
      author_email='risingodegua@gmail.com',
      url='https://github.com/risenW/datasist/tree/master/datasist/datasist',
      keywords=['Data Analysis', 'Feature Engineering', 'Visualization'],
      download_url='https://github.com/risenW/datasist/archive/v_01.tar.gz', 
      install_requires=[
        'pandas',
        'matplotlib',
        'seaborn',
        'numpy',
         ],
      classifiers=[
        'Development Status :: 3 - Alpha',   
        'Intended Audience :: Developers',      
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3.7',
      ],
      
      )
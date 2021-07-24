from setuptools import setup, find_packages

with open("README.md", "r") as cd:
  long_description = cd.read()

with open('requirements.txt') as f:
  requirements = f.readlines()

setup(
      name='datasist',
      version='1.5.3',
      license='MIT',
      description='A Python library for easy data analysis, visualization, exploration and modeling',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Rising Odegua',
      author_email='risingodegua@gmail.com',
      url='https://github.com/risenW/datasist',
      keywords=['Data Analysis', 'Feature Engineering', 'Visualization', 'project management', 'modeling', 'machine learning'],
      download_url='https://github.com/risenW/datasist/archive/v1.5.3.tar.gz', 
      packages = find_packages(),
      entry_points = {
        'console_scripts': [
          'startproject = datasist.project:startproject'
        ]
      },
      install_requires=[
        'pandas',
        'matplotlib',
        'seaborn',
        'numpy',
        'jupyter',
        'scikit-learn',
        'nltk',
        'Joblib'
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
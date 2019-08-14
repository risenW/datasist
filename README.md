<div align="center">
  <h1><b>datasist</b></h1>
</div>

-----------------

# datasist: Python library for easy modelling, visualization and exploration

## What is it?

**datasist** is a python package providing fast, quick, and a abstracted interface to 
popular and frequently used functions or techniques relating to data analysis, vusialization, data exploration,
feature engineering etc.


## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/risenW/datasist

Binary installers for the latest released version are not yet available at the [Python
package index]


## Dependencies
- Numpy
- pandas
- seaborn
- matplotlib


## Installation from sources (<b>For contributors</b>)
To install datasist from source you need python 3 in addition to the normal
dependencies above.


Clone the repo at https://github.com/risenW/datasist.git, then execute:

```sh
cd datasist
python setup.py install
```

Alternatively, you can use `pip` if you want all the dependencies pulled
in automatically (the `-e` option is for installing it in [development
mode]:

```sh
pip install -e .
```

## Documentation
No official documentation yet

### Usage
#### Jupyter Notebook

Start by loading in your pandas DataFrame, e.g. by using
```python
import pandas as pd
import datasist as ds

df = pd.read_csv('iris.csv')
```
using the structdata module, we can quickly describe the data set.

```python
ds.structdata.describe(df)
```

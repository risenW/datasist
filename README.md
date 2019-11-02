<div align="center">
  <img src='datasist.png' alt="datasist" style="width: 500px; height: 350px; margin-left: 200px;">
</div>

# datasist: Python library for easy modelling, visualization and exploration

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/datasist/">
    <img src="https://img.shields.io/badge/pip-v0.1-blue.svg" alt="latest release" />
    </a>
  </td>
</tr>
<tr>
  <td>Release Status</td>
  <td>
    <a>
    <img src="https://img.shields.io/badge/status-alpha-brightgreen.svg" alt="status" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a>
    <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="license" />
         </a>
</td>
</tr>

</table>

## What is it?

**datasist** is a python package providing fast, quick, and an abstracted interface to 
popular and frequently used functions or techniques relating to data analysis, visualization, data exploration,
feature engineering, Computer, NLP, Deep Learning, modelling, model deployment etc.

## Install
```sh
pip install datasist
```

## Dependencies
- Numpy
- pandas
- seaborn
- matplotlib


## Installation from source (<b>For contributors</b>)
To install datasist from source you need python 3.6> in addition to the normal
dependencies above. 

Run the following command in a terminal/command prompt

```sh
git clone https://github.com/risenW/datasist.git
cd datasist
python setup.py install
```

Alternatively, you can use install with `pip` after cloning, if you want all the dependencies pulled
in automatically (the `-e` option is for installing it in [development
mode]:

```sh
git clone https://github.com/risenW/datasist.git
cd datasist
pip install -e .
```

## Documentation
API documentation can be found [here](https://risenw.github.io/datasist/index.html)

### Example Usage

[Classification problem using Xente fraud dataset](https://risenw.github.io/datasist/classification_example.html)

[Basic example using the Iris dataset](https://github.com/risenW/datasist/blob/master/datasist/datasist/examples/Example_irisdata.ipynb)

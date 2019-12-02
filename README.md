<div align="center">
  <img src='datasist.png' alt="datasist" style="width: 500px; height: 350px; margin-left: 200px;">
</div>

# datasist: Python library for easy data modeling, visualization, exploration and analysis.
<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/datasist/">
    <img src="https://img.shields.io/badge/pip-v1.0-blue.svg" alt="latest release" />
    </a>
  </td>
</tr>
<tr>
  <td>Release Status</td>
  <td>
    <a>
    <img src="https://img.shields.io/badge/status-stable-brightgreen.svg" alt="status" />
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
  
<tr>
  <td>Build Status</td>
  <td>
    <a>
    <img src="https://travis-ci.org/risenW/datasist.svg?branch=master" alt="build status" />
         </a>
</td>
  
</tr>

</table>

## What is it?

**datasist** is a python package providing fast, quick, and an abstracted interface to 
popular and frequently used functions or techniques relating to data analysis, visualization, data exploration,
feature engineering, Computer, NLP, Deep Learning, modeling, model deployment etc.

## Install
```sh
pip install datasist
```

## Dependencies
- Numpy
- pandas
- seaborn
- matplotlib
- scikit-learn


## Installation from source
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

## Contributing to datasist

All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the **[contributing guide](https://risenw.github.io/datasist/contributing.html)**.

If you are simply looking to start working with the datasist codebase, navigate to the [GitHub "issues"tab](https://github.com/risenW/datasist/issues) and start looking through interesting issues. There are a number of issues listed under good first issue where you could start out.


### Example Usage

[Classification problem using Xente fraud dataset](https://risenw.github.io/datasist/classification_example.html)

[Basic example using the Iris dataset](https://github.com/risenW/datasist/blob/master/datasist/examples/Example_irisdata.ipynb)

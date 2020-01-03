import os
import shutil
from datasist.start_project import start_project


def test_start_project():
    expected = ['notebooks', 'README.txt', 'data', 'test', 'scripts', 'models']
    start_project("tests/sampletest")
    output = os.listdir("tests/sampletest/")
    assert expected == output
    # clean directory
    shutil.rmtree("tests/sampletest")



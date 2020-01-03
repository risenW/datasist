import os
import shutil
from datasist.project import start_project


def test_start_project():
    expected = ['notebooks',
                'README.txt',
                'data',
                'test',
                'scripts',
                'config.json',
                'models',
                'outputs']
    start_project("tests/sampletest")
    output = os.listdir("tests/sampletest/")
    print(output)
    assert expected == output
    # clean directory
    shutil.rmtree("tests/sampletest")



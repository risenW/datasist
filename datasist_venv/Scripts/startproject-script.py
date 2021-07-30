#!"c:\users\adesoba olamide\datasist\datasist_venv\scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'datasist','console_scripts','startproject'
__requires__ = 'datasist'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('datasist', 'console_scripts', 'startproject')()
    )

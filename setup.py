# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------
from setuptools import setup, find_packages
# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
with open('README.md', 'r') as fh:
    long_description = fh.read()
# -----------------------------------------------------------------------------
# setup
# -----------------------------------------------------------------------------
setup(
    name='vision',  
    version='0.1',
    author='Luis Monteiro',
    author_email='monteiro.lcm@gmail.com',
    description='vision detection',
    long_description=long_description,
    url='',
    packages=[
        'vision',
        'vision.bindings',
        'vision.bindings.robotframework',
    ],
    install_requires=[
        'opencv-python',
        'argparse',
        'seaborn',
        'imutils',
        'numba',
        'robotremoteserver',
        'pytesseract'
    ],
    entry_points={
      'console_scripts': [
          'vision-robot= vision.bindings.robotframework:main'
      ]
    }
 )
 # ----------------------------------------------------------------------------
 # end
 # ----------------------------------------------------------------------------

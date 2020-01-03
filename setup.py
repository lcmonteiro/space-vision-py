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
        'vision'
    ],
    install_requires=[
        'opencv-python',
        'argparse',
        'logging',
        'seaborn',
        'imutils',
        'robotremoteserver',
        'pytesseract'
    ],
    entry_points={
      'console_scripts': [
          'vision-app= applications:main'
      ]
    }
 )
 # ----------------------------------------------------------------------------
 # end
 # ----------------------------------------------------------------------------

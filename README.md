# pygert
Attempted Python 3 port of this https://github.com/bwrc/eogert

Installation
============

Labstreaming layer
------------------
PyGERT needs lab streaming layer (LSL) for data input. Current implementation uses the Python 3 version of LSL bundled in the MIDAS framework (https://github.com/bwrc/midas) so install this one first.

Other dependencies
------------------
PyGERT depends heavily on scientific libraries of Python so following libraries must be installed

- sklearn
- matlplotlib
- scipy
- numpy

Note that scipy version must be >0.16 due to an outstanding bug.

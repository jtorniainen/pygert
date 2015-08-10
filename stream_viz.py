#!/usr/bin/env python3

# Copyright 2014 Jari Torniainen <jari.torniainen@ttl.fi>
# Finnish Institute of Occupational Health
#
# This code is released under the MIT License
# http://opensource.org/licenses/mit-license.php
#
# Please see the file LICENSE for details.

from lsltools.vis import Grapher
from midas.node import lsl
import sys


def realtime_plot(stream_name='EOG'):
    s = lsl.resolve_byprop('name', stream_name)[0]
    g = Grapher(s, 500*10, 'y')
    del g

if __name__ == '__main__':
    if len(sys.argv) == 2:
        realtime_plot(sys.argv[1])
    else:
        realtime_plot()

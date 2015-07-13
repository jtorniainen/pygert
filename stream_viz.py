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

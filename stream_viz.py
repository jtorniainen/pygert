from lsltools.vis import Grapher
from midas.node import lsl


def realtime_plot():
    s = lsl.resolve_byprop('name', 'EOG')[0]
    g = Grapher(s, 500*10, 'y')

if __name__ == '__main__':
    realtime_plot()

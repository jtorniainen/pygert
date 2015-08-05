import scipy.io as io
from midas.node import lsl
import time


def start_streaming():

    data = io.loadmat('full_set.mat')
    eog_h = data['EOGh'][0]
    eog_v = data['EOGv'][0]

    info = lsl.StreamInfo('EOG', 'EOG', 2, 500, 'float32', '001')
    outlet = lsl.StreamOutlet(info)

    fs = 500.0
    idx = 0
    print('Starting stream (Ctrl+c to stop)...')
    try:
        while True:
            outlet.push_sample([eog_h[idx], eog_v[idx]])
            idx += 1
            if idx == len(eog_h):
                idx = 0
            time.sleep(1.0 / fs)

    except KeyboardInterrupt:
        print('\nStopping stream.')


if __name__ == '__main__':
    start_streaming()

from sklearn.mixture import GMM
import numpy as np


class PyGERT(object):
    def __init__(self):

        self.is_trained = False
        self.filt_len = 10

    def train(self, dh_train, dv_train):

        burn_off = 2*self.filt_len - 1

        if len(burn_off) > len(dh_train):
            print('Not enough samples!')
            return
        elif len(dh_train) != len(dv_train):
            print('Training vectors are of unequal lengths!')
            return

        dh_train = np.diff(dh_train[burn_off:])
        dv_train = np.diff(dv_train[burn_off:])

        norm_train = np.zeros((len(dh_train), 1))

        for i, (h, v) in enumerate(zip(dh_train, dv_train)):
            norm_train[i] = np.sqrt(np.dot([h, v], [h, v]))

        curr_min = norm_train[0]
        curr_max = norm_train[1]

        norm_peak = []

    def process(self, sample):
        if self.is_trained:
            return 1
        else:
            print('Train the system first!')
            return None
        pass


def test_run():
    pg = PyGERT()
    pg.train(10, 10)

if __name__ == '__main__':
    test_run()

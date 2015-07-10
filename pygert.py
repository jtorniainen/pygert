from sklearn.mixture import GMM
import numpy as np
import matplotlib.mlab as mlab
from midas.node import lsl
import scipy.signal as sig


class PyGERT(object):
    def __init__(self, stream_name='EOG'):

        self.is_trained = False

        if stream_name:  # This needs error handling!
            self._stream_handle = lsl.resolve_byprop('name', stream_name)[0]
            self._inlet = lsl.StreamInlet(self._stream_handle)
            self.srate = self._stream_handle.nominal_srate()

        # Configure filters
        self.filt_len = 150
        stop_limit = 500 / self.srate
        wn1 = stop_limit / (self.srate/2)
        self._filt_coef_b1 = sig.firwin(self.filt_len, wn1)

    def run_training(self, train_time_s=60):

        n = round(train_time_s * self.srate)
        train_eog = np.ndarray((0, 2))

        print('Collecting %d seconds of data for training...' % train_time_s)

        while train_eog.shape[0] < n:
            chnk, t = self._inlet.pull_chunk()
            if chnk:
                train_eog = np.vstack((train_eog, np.asarray(chnk)))
            print('\t%d%%\r' % (100 * (train_eog.shape[0] / n)), end="")

        train_eog = sig.filtfilt(self._filt_coef_b1, 1, train_eog, axis=0)

        print('Data collected. Training classifier...')

        self._train(train_eog[:, 0], train_eog[:, 1])


    def _train(self, dh_train, dv_train):
        """ Train the classifier. """
        burn_off = 2*self.filt_len - 1

        if burn_off > len(dh_train):
            print('Not enough samples!')
            return
        elif len(dh_train) != len(dv_train):
            print('Training vectors are of unequal lengths!')
            return

        # --- STEP 1 ---
        dh_train = np.diff(dh_train[burn_off:])
        dv_train = np.diff(dv_train[burn_off:])

        # --- STEP 2 ---
        norm_train = np.zeros((len(dh_train), 1))
        for i, (h, v) in enumerate(zip(dh_train, dv_train)):
            norm_train[i] = np.sqrt(np.dot([h, v], [h, v]))

        curr_max = norm_train[0]
        curr_min = norm_train[0]

        norm_peak = []
        for x in norm_train:  # TODO: check if this loop can be truncated
            if x > curr_max:
                curr_max = x
                curr_min = curr_max
            if x < curr_min:
                curr_min = x
            else:
                if curr_max > curr_min:
                    norm_peak.append(curr_max[0])
                curr_max = curr_min

        if len(norm_peak) < 2:
            print('Not enough peaks (>=2) in the norm-vector. Aborting')
            return

        # --- STEP 3 ---
        g = GMM(n_components=2, n_iter=100)
        g.set_params(init_params='wc')

        norm_peak = np.sort(norm_peak)
        norm_peak = norm_peak.reshape(norm_peak.shape[0], 1)

        init_vals = np.ndarray((2, 1))
        init_vals[0] = norm_peak[0]
        init_vals[1] = norm_peak[-1]
        g.means_ = init_vals

        g.fit(norm_peak)  # Check if dimension mattered here

        self.mu_fix = g.means_[0]
        self.sigma_fix = np.sqrt(g.covars_[0])
        self.prior_fix = g.weights_[0]

        self.mu_bs = g.means_[1]
        self.sigma_bs = np.sqrt(g.covars_[1])
        self.prior_bs = g.weights_[1]

        # --- STEP 4 ---
        curr_max = dv_train[0]
        curr_min = dv_train[0]
        diff_max_min = []
        time_diff = []
        CURR_MIN = []
        CURR_MAX = []
        NTR = []
        tmp_i = 1

        for idx, dv in enumerate(dv_train):
            if dv > curr_max:
                curr_max = dv
                curr_min = curr_max
                tmp_i = idx
            if dv < curr_min:
                curr_min = dv
            else:
                if curr_max > curr_min:
                    ntr = norm_train[tmp_i]
                    p_bs = (mlab.normpdf(ntr, self.mu_bs, self.sigma_bs) *
                            self.prior_bs /
                            (mlab.normpdf(ntr, self.mu_bs, self.sigma_bs) *
                             self.prior_bs +
                             mlab.normpdf(ntr, self.mu_fix, self.sigma_fix) *
                             self.prior_fix))

                    if p_bs > 2/3:
                        feature = curr_max - curr_min - abs(curr_max + curr_min)
                        if feature > 0:
                            diff_max_min.append(feature)
                            time_diff.append(idx)
                            CURR_MAX.append(curr_max)
                            CURR_MIN.append(curr_min)
                            NTR.append(ntr)

                curr_max = curr_min

        # --- STEP 5 ---
        if len(diff_max_min) < 2:
            print('Aborting')
            return

        diff_max_min = np.sort(diff_max_min)
        diff_max_min = diff_max_min.reshape(diff_max_min.shape[0], 1)

        init_vals = np.ndarray((2, 1))
        init_vals[0] = diff_max_min[0]
        init_vals[1] = diff_max_min[-1]
        g.means_ = init_vals

        g.fit(diff_max_min)

        self.mu_sac = g.means_[0]
        self.sigma_sac = np.sqrt(g.covars_[0])
        self.prior_sac = g.weights_[0]

        self.mu_bli = g.means_[1]
        self.sigma_bli = np.sqrt(g.covars_[1])
        self.prior_bli = g.weights_[1]

        if self.sigma_sac == 0 or self.sigma_bli == 0 or self.sigma_fix == 0:
            print('Zero variance detected! Aborting...')
            return

        self.is_trained = True
        self.just_trained = True

    def run_detection(self):
        if self.is_trained:

            # 1. Clear lsl queue
            # 2. Read new sample
            # 3. Filter sample
            # 4.
            return 1
        else:
            print('Train the system first!')
            return None
        pass


def test_run():
    pg = PyGERT()
    pg.run_training()

if __name__ == '__main__':
    test_run()

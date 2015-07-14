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
        filt_len = 150
        stop_limit = 500 / self.srate
        wn1 = stop_limit / (self.srate/2)

        self._b1 = sig.firwin(filt_len, wn1)
        self._z1_h = sig.lfilter_zi(self._b1, 1)
        self._z1_v = sig.lfilter_zi(self._b1, 1)

        self._eog_h_prev = 0
        self._eog_v_prev = 0

        # Experimental dbg outlet
        info = lsl.StreamInfo('dbg', 'dbg', 2, 500, 'float32', '003')
        self.outlet = lsl.StreamOutlet(info)

    def run_training(self, train_time_s=60):

        n = round(train_time_s * self.srate)
        train_eog = np.ndarray((0, 2))

        print('Collecting %d seconds of data for training...' % train_time_s)

        while train_eog.shape[0] < n:
            chnk, t = self._inlet.pull_chunk()
            if chnk:
                train_eog = np.vstack((train_eog, np.asarray(chnk)))
            # print('\t%d%%\r' % (100 * (train_eog.shape[0] / n)), end="")

        train_eog = sig.filtfilt(self._b1, 1, train_eog, axis=0)

        print('Data collected. Training classifier...')

        training_ok = self._train(train_eog[:, 0], train_eog[:, 1])
        if training_ok:
            print('Training finished ok.')
        else:
            print('Training had errors, re-run.')

    def _train(self, dh_train, dv_train):
        """ Train the classifier. """

        dh_train = np.diff(dh_train)
        dv_train = np.diff(dv_train)

        if len(dh_train) != len(dv_train):
            print('Training vectors are of unequal lengths!')
            return False

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
            return False

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

                curr_max = curr_min

        # --- STEP 5 ---
        if len(diff_max_min) < 2:
            print('Not enough diff_max_min samples! Aborting')
            return False

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

        if np.equal(0, [self.sigma_sac, self.sigma_bli, self.sigma_bs,
                        self.sigma_fix]).any():
            print('Zero variance detected! Aborting...')
            return False

        self.is_trained = True
        self._just_trained = True
        return True

    def _get_sample(self):
        sample, t = self._inlet.pull_sample(timeout=0)
        if sample:
            eog_h = float(sample[0])
            eog_v = float(sample[1])
            eog_h, self._z1_h = sig.lfilter(self._b1, 1, [eog_h], zi=self._z1_h)
            eog_v, self._z1_v = sig.lfilter(self._b1, 1, [eog_v], zi=self._z1_v)
            return eog_h, eog_v
        else:
            return None, None

    def _filter_sample(self, eog_h, eog_v):
        eog_h, self._z1_h = sig.lfilter(self._b1, 1, [eog_h], zi=self._z1_h)
        eog_v, self._z1_v = sig.lfilter(self._b1, 1, [eog_v], zi=self._z1_v)
        return eog_h, eog_v

    def _clear_lsl_queue(self):
        while self._inlet.pull_sample(timeout=0.0)[0]:
            pass

    def _norm(self, x):
        return np.sqrt(np.dot(x, x))

    def _detect(self, eog_h, eog_v):

        diff_h = eog_h - self._eog_h_prev
        diff_v = eog_v - self._eog_v_prev

        self._eog_h_prev = eog_h
        self._eog_v_prev = eog_v

        if self._just_trained:
            self.curr_vd_max = diff_v
            self._diff_v_prev = diff_v
            self._just_trained = False

        if diff_v > self._diff_v_prev:
            self.curr_vd_max = diff_v
        self._diff_v_prev = diff_v
        dmm = self.curr_vd_max - diff_v - abs(self.curr_vd_max + diff_v)

        norm_d1 = self._norm([float(diff_h), float(diff_v)])

        # norm_d2 = self._norm([]) do this later
        # Some buffer stuff here with 2s (later)

        # Compute likelihoods using asymmetric distributions.
        if norm_d1 > self.mu_fix:  # likelihood of the fixation point
            lh_fix = mlab.normpdf(norm_d1, self.mu_fix, self.sigma_fix)
        else:
            lh_fix = mlab.normpdf(self.mu_fix, self.mu_fix, self.sigma_fix)

        if norm_d1 < self.mu_bs:  # likelihood of the blink-or-saccade
            lh_bs = mlab.normpdf(norm_d1, self.mu_bs, self.sigma_bs)
        else:
            lh_bs = mlab.normpdf(self.mu_bs, self.mu_bs, self.sigma_bs)

        if dmm > self.mu_sac:   # likelihood (dmm) of the saccade
            lh_sac = mlab.normpdf(dmm, self.mu_sac, self.sigma_sac)
        else:
            lh_sac = mlab.normpdf(self.mu_sac, self.mu_sac, self.sigma_sac)

        if dmm < self.mu_bli:  # likelihood (dmm) of the blink
            lh_bli = mlab.normpdf(dmm, self.mu_bli, self.sigma_bli)
        else:
            lh_bli = mlab.normpdf(self.mu_bli, self.mu_bli, self.sigma_bli)

        # Compute the posterior probabilities
        # evidence of norm data, i.e. the normalization constant
        evi_norm = lh_fix * self.prior_fix + lh_bs * self.prior_bs
        # normalized probability for the sample to be a fixation point
        pfn = lh_fix * self.prior_fix / evi_norm
        # normalized probability for the sample to be a saccade or blink
        psbn = 1 - pfn
        # evidence (dmm), i.e. the normalization constant
        evi_dmm = lh_sac * self.prior_sac + lh_bli * self.prior_bli
        # normalized probability for the sample to be a blink
        pbn = psbn * lh_bli * self.prior_bli / evi_dmm
        # normalized probability for the sample to be a saccade
        psn = psbn * lh_sac * self.prior_sac / evi_dmm
        print('\t%0.2f\t%0.2f\t%0.2f' % (pfn, pbn, psn))
        return [float(pfn), float(psn), float(pbn)]

    def run_detection(self):
        print('Starting online EOG event detection (Ctrl+c to quit)')
        self._clear_lsl_queue()  # Clear the incoming buffer
        if self.is_trained:
            try:
                while True:
                    # 1. Read new sample
                    eog_h, eog_v = self._get_sample()
                    # 2. Perform detection
                    if eog_h and eog_v:
                        self.outlet.push_sample([eog_h, eog_v])
                        self._detect(eog_h, eog_v)

            except KeyboardInterrupt:
                print('\nDetection stopped.')
        else:
            print('Train the system first!')
            return None


def test_run():
    pg = PyGERT()
    pg.run_training()
    input('Pausing here (ENTER to continue)')
    pg.run_detection()

if __name__ == '__main__':
    test_run()

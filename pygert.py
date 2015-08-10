#!/usr/bin/env python3

# Copyright 2014 Jari Torniainen <jari.torniainen@ttl.fi>
# Finnish Institute of Occupational Health
#
# This code is released under the MIT License
# http://opensource.org/licenses/mit-license.php
#
# Please see the file LICENSE for details.

from sklearn.mixture import GMM
import numpy as np
import matplotlib.mlab as mlab
from midas.node import lsl
import scipy.signal as sig
import time


class PyGERT(object):

    def __init__(self, stream_name='EOG', srate=500):
        """ Initialize the PyGERT-object

        Args:
            stream_name <string>: name of the lsl stream to connect to
        """

        self._is_trained = False

        if stream_name:  # This needs error handling!
            self._stream_handle = lsl.resolve_byprop('name', stream_name)[0]
            self._inlet = lsl.StreamInlet(self._stream_handle)
            self.srate = self._stream_handle.nominal_srate()
        else:
            self.srate = srate  # Fix this later

        # Configure filters
        filt_len = 150
        stop_limit1 = 500 / self.srate
        stop_limit2 = 40
        wn1 = stop_limit1 / (self.srate/2)
        wn2 = stop_limit2 / (self.srate/2)

        self._group_delay = (filt_len - 1) / 2

        self._b1 = sig.firwin(filt_len, wn1)
        self._b2 = sig.firwin(filt_len, wn2)
        self._z1_h = sig.lfilter_zi(self._b1, 1)
        self._z1_v = sig.lfilter_zi(self._b1, 1)
        self._z2_h = sig.lfilter_zi(self._b2, 1)
        self._z2_v = sig.lfilter_zi(self._b2, 1)

        self._eog_h_prev1 = 0
        self._eog_v_prev1 = 0
        self._eog_h_prev2 = 0
        self._eog_v_prev2 = 0

        # Buffers
        self._MAX_SACCADE_LENGTH = .2
        self._MAX_BLINK_LEN = .5
        self._MIN_SACCADE_LEN = .01
        self._MIN_BLINK_LEN = .2
        self._MIN_SACCADE_GAP = .1

        buffer_len_saccade = round(self._MAX_SACCADE_LENGTH * self.srate)
        buffer_len_blink = round(self._MAX_BLINK_LEN * self.srate)
        self._norm_d2 = np.zeros(buffer_len_saccade)
        self._eog_v2 = np.zeros(buffer_len_blink)
        self._psn = np.zeros(buffer_len_saccade)

        self._fix_prob_mass = 0

        self._saccade_samples = 0
        self._saccade_prob = 0

        self._blink_samples = 0
        self._blink_prob = 0

        self._n = 0
        self._n_sac = 0
        self._n_blink = 0

        self._SAC_START = []
        self._SAC_END = []
        self._SAC_DUR = []
        self._SAC_PROB = []

        self._BLI_START = []
        self._BLI_END = []
        self._BLI_DUR = []
        self._BLI_PROB = []

        # Flags
        self._blink_on = False
        self._saccade_on = False

    def run_training(self, train_time_s=60):
        """ Runs the training with live data.

        Args:
            train_time_s <float>: size of the training segment in seconds.
        """

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

    def _find_norm_peaks(self, norm_train):
        """ Find peaks in the input vector.

        Args:
            norm_train <array_like>: a vector
        """
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
        return(norm_peak)

    def _find_diff_max_min(self, dv_train, norm_train):
        """ Find max-min values of dv peaks

        Args:
            dv_train <array_like>: vector
            norm_train <array_like: vector
        """
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
        return diff_max_min

    def _train(self, dh_train, dv_train):
        """ Trains the probabilistic classifier.

        Args:
            dh_train <array_like>: horizontal EOG vector
            dv_train <array_like>: vertical EOG vector
        Returns:
            training_succesful <boolean>: status of training
        """

        dh_train = np.diff(dh_train)
        dv_train = np.diff(dv_train)

        if len(dh_train) != len(dv_train):
            print('Training vectors are of unequal lengths!')
            return False

        # --- STEP 2 ---
        norm_train = np.zeros((len(dh_train), 1))
        for i, (h, v) in enumerate(zip(dh_train, dv_train)):
            norm_train[i] = np.sqrt(np.dot([h, v], [h, v]))

        norm_peak = self._find_norm_peaks(norm_train)

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
        diff_max_min = self._find_diff_max_min(dv_train, norm_train)

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

        self._is_trained = True
        self._just_trained = True
        return True

    def _get_sample(self):
        """ Pulls and filters a single sample from the lsl stream. """
        sample, t = self._inlet.pull_sample(timeout=0)
        if sample:
            self._n += 1
            eog_h = [float(sample[0])]
            eog_v = [float(sample[1])]
            eog_h1, self._z1_h = sig.lfilter(self._b1, 1, eog_h, zi=self._z1_h)
            eog_v1, self._z1_v = sig.lfilter(self._b1, 1, eog_v, zi=self._z1_v)
            eog_h2, self._z2_h = sig.lfilter(self._b2, 1, eog_h, zi=self._z2_h)
            eog_v2, self._z2_v = sig.lfilter(self._b2, 1, eog_v, zi=self._z2_v)
            return eog_h1, eog_v1, eog_h2, eog_v2
        else:
            return None, None, None, None

    def _filter_sample(self, eog_h, eog_v):
        """ Filters an EOG sample using the internal FIR-filter. """
        # This function is used for testing, do not remove!
        eog_h1, self._z1_h = sig.lfilter(self._b1, 1, eog_h, zi=self._z1_h)
        eog_v1, self._z1_v = sig.lfilter(self._b1, 1, eog_v, zi=self._z1_v)
        eog_h2, self._z2_h = sig.lfilter(self._b2, 1, eog_h, zi=self._z2_h)
        eog_v2, self._z2_v = sig.lfilter(self._b2, 1, eog_v, zi=self._z2_v)
        return eog_h1, eog_v1, eog_h2, eog_v2

    def _clear_lsl_queue(self):
        """ Empty the LSL buffer. """
        while self._inlet.pull_sample(timeout=0.0)[0]:
            pass

    def _norm(self, x):
        """ Calculates the norm of input vector

        Args:
            x <array_like>: Input vector
        """
        return np.sqrt(np.dot(x, x))

    def _detect(self, eog_h1, eog_v1, eog_h2, eog_v2):
        """ Runs the detection for a single sample

        Args:
            eog_h <float>: Horizontal EOG sample
            eog_v <float>: Vertical EOG sample
        Returns:
            probabilities <list>: List of probabilities for sample belonging to
                                  one of the events [fixation, saccade, blink]
        """

        self._n += 1

        diff_h1 = eog_h1 - self._eog_h_prev1
        diff_v1 = eog_v1 - self._eog_v_prev1

        diff_h2 = eog_h2 - self._eog_h_prev2
        diff_v2 = eog_v2 - self._eog_v_prev2

        self._eog_h_prev1 = eog_h1
        self._eog_v_prev1 = eog_v1

        self._eog_h_prev2 = eog_h2
        self._eog_v_prev2 = eog_v2

        if self._just_trained:
            self.curr_vd_max = diff_v1
            self._diff_v_prev = diff_v1
            self._just_trained = False

        self.curr_vd_max = max([self.curr_vd_max, diff_v1])

        self._diff_v_prev = diff_v1
        dmm = self.curr_vd_max - diff_v1 - abs(self.curr_vd_max + diff_v1)

        norm_d1 = self._norm([float(diff_h1), float(diff_v1)])
        norm_d2 = self._norm([float(diff_h2), float(diff_v2)])

        # Some buffer stuff here with 2s (later)
        self._norm_d2 = np.roll(self._norm_d2, -1)
        self._norm_d2[-1] = norm_d2

        self._eog_v2 = np.roll(self._eog_v2, -1)
        self._eog_v2[-1] = eog_v2

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

        # Update saccade probability buffer
        self._psn = np.roll(self._psn, -1)
        self._psn[-1] = psn

        self._fix_prob_mass += pfn
        if psn > max([pfn, pbn]):
            self._saccade_on = True
            self._saccade_samples += 1
            self._saccade_prob = self._saccade_prob + psn
        elif self._saccade_on:
            self._saccade_terminated()

        if pbn > max([pfn, psn]):
            self._blink_on = True
            self._blink_samples += 1
            self._blink_prob += pbn

        elif self._blink_on:
            self._blink_terminated()
        return [float(pfn), float(psn), float(pbn)]

    def _blink_terminated(self):
        """ Compute the blink duration and starting times """
        if self._blink_prob > self._MIN_BLINK_LEN / 4 * self.srate:
            buffer_dvf2 = np.diff(self._eog_v2)
            peak_val_dvf2 = max(buffer_dvf2)
            peak_idx_dvf2 = np.argmax(buffer_dvf2)
            for peak_start in np.arange(peak_idx_dvf2, 0, -1):
                if buffer_dvf2[peak_start] < .1 * peak_val_dvf2:
                    break

            # calculate an estimated blink duration (in samples) assuming
            # symmetric peak
            peak_idx_vf2 = np.argmax(self._eog_v2)
            blink_dur = 2 * (peak_idx_vf2 - peak_start)

            # Final check to determine if this is an actual blink
            if blink_dur > self._MIN_BLINK_LEN * self.srate:
                # This is a real blink
                self._n_blink += 1
                buffer_start = round(max(1, self._n - self._MAX_BLINK_LEN *
                                         self.srate))
                self._BLI_START.append((buffer_start + peak_start -
                                        self._group_delay) / self.srate)
                self._BLI_DUR.append(blink_dur / self.srate)
                self._BLI_PROB.append(self._blink_prob / self._blink_samples)

            # Blinks form a sac-blink-sac sequence so remove the most recent
            # saccade (the first "sac" in the sequence) if they overlap as it
            # was not "real" saccade
            if self._n_sac > 0 and self._n_blink > 0:
                if self._BLI_START[-1] < (self._SAC_START[-1] +
                                          self._SAC_DUR[-1]):
                    self._SAC_START.pop()
                    self._SAC_DUR.pop()
                    self._SAC_PROB.pop()
                    self._n_sac -= 1

        self._blink_on = False
        self._blink_samples = 0
        self._blink_prob = 0

    def _find_peak(self):
        """ Utility function to find the first and last index of the peak. """
        peak_idx = np.argmax(self._norm_d2)
        if peak_idx == 0:
            start_idx = 0
        else:
            for start_idx in np.arange(peak_idx, 0, -1):
                if self._norm_d2[start_idx] - self._norm_d2[start_idx + 1] > 0:
                    break
            start_idx += 1

        if peak_idx == self._norm_d2.size - 1:
            end_idx = self._norm_d2.size - 1
        else:
            for end_idx in range(peak_idx + 1, self._norm_d2.size):
                if self._norm_d2[end_idx] - self._norm_d2[end_idx - 1] > 0:
                    break

        end_idx -= 1

        return start_idx, end_idx

    def _saccade_terminated(self):
        """ Process a saccade that just ended. """
        print('%d SACCADE ENDED?' % time.time())
        if self._saccade_prob > self._MIN_SACCADE_LEN * self.srate:
            peak_start_idx, peak_end_idx = self._find_peak()

            saccade_dur = peak_end_idx - peak_start_idx
            saccade_prob_mass = sum(self._psn[round(peak_start_idx):-1])
            buf_start = round(max(1, self._n - self._norm_d2.size))
            saccade_start_n = buf_start + peak_start_idx - self._group_delay - 1

            saccade_ok = True
            if saccade_prob_mass < self._MIN_SACCADE_LEN * self.srate:
                saccade_ok = False

            if self._n_blink > 0:
                if saccade_start_n / self.srate < (self._BLI_START[-1] +
                                                   self._BLI_DUR[-1]):
                    saccade_ok = False

            if self._n_sac > 0:
                if self._fix_prob_mass < self._MIN_SACCADE_GAP * self.srate:
                    saccade_ok = False

                saccade_gap = (saccade_start_n / self.srate -
                               (self._SAC_START[-1] + self._SAC_DUR[-1]))

                if saccade_gap < self._MIN_SACCADE_GAP * self.srate:
                    saccade_ok = False

            if saccade_ok:
                self._n_sac += 1
                self._SAC_START.append(saccade_start_n / self.srate)
                self._SAC_DUR.append(saccade_dur / self.srate)
                self._SAC_PROB.append(self._saccade_prob /
                                      self._saccade_samples)
                self._fix_prob_mass = 0
                self._sac_on_end_prev_n = self._n

        self._saccade_on = False
        self._saccade_samples = 0
        self._saccade_prob = 0

    def run_detection(self):
        """ Starts the online detection of EOG events. """

        print('Starting online EOG event detection (Ctrl+c to quit)')

        self._clear_lsl_queue()  # Clear the incoming buffer
        if self._is_trained:
            try:
                while True:
                    # 1. Read next sample
                    eog_h1, eog_v1, eog_h2, eog_v2 = self._get_sample()
                    # 2. Perform detection and print probabilities
                    if eog_h1 and eog_v1:
                        pr = self._detect(eog_h1, eog_v1, eog_h2, eog_v2)
                        print('\t%0.2f\t%0.2f\t%0.2f' % (pr[0], pr[1], pr[2]))
            except KeyboardInterrupt:
                print('\nDetection stopped.')
        else:
            print('Train the system first!')


def test_run():
    """ Some tests. """
    pg = PyGERT()
    pg.run_training(train_time_s=60)
    input('Pausing here (ENTER to continue)')
    pg.run_detection()

if __name__ == '__main__':
    test_run()

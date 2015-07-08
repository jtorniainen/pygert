from sklearn.mixture import GMM
import numpy as np
import matplotlib.mlab as mlab


class PyGERT(object):
    def __init__(self):

        self.is_trained = False
        self.filt_len = 10

    def train(self, dh_train, dv_train):
        """ Train the classifier. """
        burn_off = 2*self.filt_len - 1

        if len(burn_off) > len(dh_train):
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
                    norm_peak.append(curr_max)
                curr_max = curr_min

        if len(norm_peak) < 2:
            print('Not enough peaks (>=2) in the norm-vector. Aborting')
            return

        # --- STEP 3 ---
        norm_peak = np.sort(norm_peak)
        g = GMM(n_components=2, n_iter=100)
        g.set_params(init_params='wc')
        g.means_ = np.asarray([norm_peak[0], norm_peak[1]])  # Initial values
        g.fit(norm_peak)  # Check if dimension mattered here

        mu_fix = g.means_[1]
        sigma_fix = np.sqrt(g.covars_[1])
        prior_fix = g.weights_[1]

        mu_bs = g.means_[0]
        sigma_bs = np.sqrt(g.covars_[0])
        prior_bs = g.weights_[0]

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
                    p_bs = (mlab.normpdf(ntr, mu_bs, sigma_bs) * prior_bs /
                            (mlab.normpdf(ntr, mu_bs, sigma_bs) * prior_bs +
                             mlab.normpdf(ntr, mu_fix, sigma_fix) * prior_fix))
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
        g.means_ = np.asarray([diff_max_min[0], diff_max_min[1]])
        g.fit(diff_max_min)

        self.mu_sac = g.means_[1]
        self.sigma_sac = np.sqrt(g.covars_[1])
        self.prior_sac = g.weights_[1]

        self.mu_bli = g.means_[0]
        self.sigma_bli = np.sqrt(g.covars_[0])
        self.prior_bli = g.weights_[0]

        if self.sigma_sac == 0 or self.sigma_bli == 0 or self.sigma_fix == 0:
            print('Zero variance detected! Aborting...')
            return

        self.is_trained = True
        self.just_trained = True

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

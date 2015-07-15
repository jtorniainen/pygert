import scipy.io as io
import scipy.signal as sig
from pygert import PyGERT
import numpy as np


def offline_test_package(train_file='train.mat', detect_file='online.mat'):
    pg = PyGERT()

    data = io.loadmat(train_file)
    eog_h = data['EOGh'][0]
    eog_v = data['EOGh'][0]
    eog_h = sig.filtfilt(pg._b1, 1, eog_h)
    eog_v = sig.filtfilt(pg._b1, 1, eog_v)
    training_ok = pg._train(eog_h, eog_v)

    if training_ok:
        print('Training finished ok.')
    else:
        print('Training had errors, re-run.')

    # Save training params to file as well?
    print('   \tmu\tsd\tpri')
    print('sac:\t%0.2f\t%0.2f\t%0.2f' % (pg.mu_sac, pg.sigma_sac,
                                         pg.prior_sac))
    print('bli:\t%0.2f\t%0.2f\t%0.2f' % (pg.mu_bli, pg.sigma_bli,
                                         pg.prior_bli))
    print('fix:\t%0.2f\t%0.2f\t%0.2f' % (pg.mu_fix, pg.sigma_fix,
                                         pg.prior_fix))
    print('bs :\t%0.2f\t%0.2f\t%0.2f' % (pg.mu_bs,  pg.sigma_bs,
                                         pg.prior_bs))

    data = io.loadmat(detect_file)
    eog_h = data['EOGh'][0]
    eog_v = data['EOGv'][0]

    results = np.ndarray((0, 3))
    for h, v in zip(eog_h, eog_v):
        h, v = pg._filter_sample(h, v)
        p = pg._detect(h, v)
        results = np.vstack((results, p))
    np.savetxt('pygert_results.csv', results, delimiter=',')


def prepare_data():
    dat = io.loadmat('test001.mat')

    eog_v = dat['EOGv'][0]
    eog_h = dat['EOGh'][0]
    fs = float(dat['fs'][0])

    fir_len = 150
    stop_limit = 500 / fs
    wn = stop_limit / (fs/2)
    b = sig.firwin(fir_len, wn)

    eog_v_f = sig.filtfilt(b, 1, eog_v)
    eog_h_f = sig.filtfilt(b, 1, eog_h)

    return eog_h_f, eog_v_f


def test_train_fulldata():
    eog_h_f, eog_v_f = prepare_data()

    p = PyGERT()
    p.train(eog_h_f, eog_v_f)
    print('   \tmu\tsd\tpri')
    print('sac:\t%0.2f\t%0.2f\t%0.2f' % (p.mu_sac, p.sigma_sac, p.prior_sac))
    print('bli:\t%0.2f\t%0.2f\t%0.2f' % (p.mu_bli, p.sigma_bli, p.prior_bli))
    print('fix:\t%0.2f\t%0.2f\t%0.2f' % (p.mu_fix, p.sigma_fix, p.prior_fix))
    print('bs :\t%0.2f\t%0.2f\t%0.2f' % (p.mu_bs,  p.sigma_bs,  p.prior_bs))


def test_train_segments():
    results = np.zeros((6, 12))
    for i in range(6):
        dat = io.loadmat('testset%02d.mat' % (i + 1))
        eog_v = dat['eog_v_seg'][0]
        eog_h = dat['eog_h_seg'][0]
        fs = float(dat['fs'][0])

        fir_len = 150
        stop_limit = 500 / fs
        wn = stop_limit / (fs/2)
        b = sig.firwin(fir_len, wn)

        eog_v_f = sig.filtfilt(b, 1, eog_v)
        eog_h_f = sig.filtfilt(b, 1, eog_h)
        p = PyGERT()
        p.train(eog_h_f, eog_v_f)

        results[i,  0] = p.mu_fix
        results[i,  1] = p.sigma_fix
        results[i,  2] = p.prior_fix
        results[i,  3] = p.mu_sac
        results[i,  4] = p.sigma_sac
        results[i,  5] = p.prior_sac
        results[i,  6] = p.mu_bli
        results[i,  7] = p.sigma_bli
        results[i,  8] = p.prior_bli
        results[i,  9] = p.mu_bs
        results[i, 10] = p.sigma_bs
        results[i, 11] = p.prior_bs

    print(results)
    np.savetxt('segment_test_python.csv', results, delimiter=',')

if __name__ == '__main__':
    # test_train_fulldata()
    # test_train_segments()
    offline_test_package()

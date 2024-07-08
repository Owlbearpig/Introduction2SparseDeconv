import timeit

from implementation import ista_deconvolve
from pathlib import Path
import numpy as np
import multiprocessing
from scipy.linalg import toeplitz, norm
from scipy.fftpack import rfft
from numpy.fft import rfft, irfft

base_dir = Path(r"/home/ftpuser/ftp/Data/Misc/MariaLisa/Baby/CompleteAreas")

test_mode = True  # enable(False) or disable(True) saving
start_ind = 0
end_ind = 2000
lambda_ = 12
step_scale = 2
max_iterations = 30


def shrinkage_factor(H):
    H = np.array(H, dtype=np.float32)
    H = np.dot(H, H.T)
    return 1 / np.max(rfft(H[0]))  # since H is circulant -> H.T @ H is -> eig vals with fft


def toeplitz_j(ref):  # convolve(ref, y, mode="wrap") from scipy.ndimage.filters
    c = np.hstack((ref, ref[0]))
    c = toeplitz(c, c[-1::-1])
    return c[0:-1, 0:-1]


def eval_area():
    # meas_sets = ["before", "after"]
    meas_sets = ["before"]
    for meas_set in meas_sets:
        data_arr_path = base_dir / f"{meas_set}.npz"
        ref_path = base_dir / f"ref_{meas_set}.txt"
        bkg_path = base_dir / f"bkg_{meas_set}.txt"

        data = np.load(str(data_arr_path))

        t_axes, y_axes, filenames = data["arr_0"], data["arr_1"], data["arr_2"]
        ref_td, bkg_td = np.loadtxt(str(ref_path)), np.loadtxt(str(bkg_path))

        def pre_process(t_, y_):
            y_ = (y_ - bkg_td[:, 1])[start_ind:end_ind]
            y_ -= np.mean(y_)

            normalization = 1 / np.max(y_)

            return t_[start_ind:end_ind], y_ * normalization

        t_ref, y_ref_corrected = pre_process(ref_td[:, 0], ref_td[:, 1])

        H = toeplitz_j(y_ref_corrected)
        tau = step_scale * shrinkage_factor(H)

        f_results = np.zeros((len(filenames), len(t_ref)))
        for i, (t, y, filename) in enumerate(zip(t_axes, y_axes, filenames)):
            t_new, y_corrected = pre_process(t, y)

            def soft_threshold(v):
                ret = np.zeros(v.shape)

                id_smaller = v <= -lambda_ * tau
                id_larger = v >= lambda_ * tau
                ret[id_smaller] = v[id_smaller] + lambda_ * tau
                ret[id_larger] = v[id_larger] - lambda_ * tau

                return ret

            cir_dot = lambda x, y: irfft(np.multiply(rfft(x[:, 0]), rfft(y)))

            def calc_cir(H_, sample_, f_):
                return cir_dot(H_.T, cir_dot(H_, f_) - sample_)

            f = np.zeros(len(t_ref))
            for n in range(max_iterations):
                pre_calc = calc_cir(H, y_corrected, f)
                f = soft_threshold(f - tau * pre_calc)

            f_results[i] = f


if __name__ == '__main__':
    t0 = timeit.default_timer()
    eval_area()
    print(timeit.default_timer() - t0)

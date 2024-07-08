import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz, norm
from scipy.fftpack import rfft
from numpy.fft import rfft, irfft
from pathlib import Path
from THz.preprocessing import butter_bandpass_filter

min_sample_points = 0
max_sample_points = 2000


def load_and_correction(path, bkg_path):
    bkg_data = np.loadtxt(bkg_path, usecols=(0, 1))
    data = np.loadtxt(path, usecols=(0, 1))
    t = data.T[0]

    end_ind = max_sample_points  # len(data.T[1])
    start_ind = min_sample_points
    fs = 1/np.mean(np.diff(t))
    y = butter_bandpass_filter((data.T[1] - bkg_data.T[1])[start_ind:end_ind], lowcut=0.05, highcut=1.8, fs=fs, order=5)
    y -= np.mean(y)

    normalization = 1 / np.max(y)



    return t[start_ind:end_ind], y * normalization


def shrinkage_factor(H):
    H = np.array(H, dtype=np.float32)
    H = np.dot(H, H.T)
    return 1 / max(rfft(H[0]))  # since H is circulant -> H.T @ H is -> eig vals with fft


def toeplitz_j(ref):  # convolve(ref, y, mode="wrap") from scipy.ndimage.filters
    c = np.hstack((ref, ref[0]))
    c = toeplitz(c, c[-1::-1])
    return c[0:-1, 0:-1]


def ista_deconvolve(ref, sample, new_imp=False, lambda_=None, step_scale=None, max_iterations=None):
    """
    sparse deconvolution using an iterative shrinkage algorithm
    :param ref: reference data array
    :param sample: sample data array
    :param tau: iteration step size
    :return: f: sparse deconvolution array,
    relerr: 1-norm(sum abs(f)) of f compared to previous iteration,
    n: number of iterations
    new_imp: Hehe
    """
    if not lambda_:
        lambda_ = 12
    if not step_scale:
        step_scale = 2

    eps = 1e-10
    if not max_iterations:
        max_iterations = 2000

    H = toeplitz_j(ref)
    tau = step_scale * shrinkage_factor(H)

    def soft_threshold(v):
        ret = np.zeros(v.shape)

        id_smaller = v <= -lambda_ * tau
        id_larger = v >= lambda_ * tau
        ret[id_smaller] = v[id_smaller] + lambda_ * tau
        ret[id_larger] = v[id_larger] - lambda_ * tau

        return ret

    def calc_mm(H_, sample_, f):
        return np.dot(H_.T, np.dot(H_, f) - sample_)

    cir_dot = lambda x, y: irfft(np.multiply(rfft(x[:, 0]), rfft(y)))

    def calc_cir(H_, sample_, f_):

        return cir_dot(H_.T, cir_dot(H_, f_) - sample_)

    if new_imp:
        calc = calc_cir
    else:
        calc = calc_mm

    # init variables
    relerr = 1
    n = 0

    f = np.zeros(ref.shape)
    ssq = 0
    big_f = []

    rel_errors = []
    # while relerr > eps and n < max_iterations:
    while n < max_iterations:
        n += 1
        pre_calc = calc(H, sample, f)
        f = soft_threshold(f - tau * pre_calc)
        # big_f.append(0.5 * norm(sample - cir_dot(H, f), 2) ** 2 + lambda_ * norm(f, 1))
        # print("F(f):", big_f[-1])
        if n - 2 == max_iterations:
            ssq = norm(f, 1)

    ssq_new = norm(f, 1)
    relerr = abs(1 - ssq / ssq_new)
    if not new_imp:
        if not n % (max_iterations // 3):
            print("relerr:", round(relerr, int(-np.log10(relerr)) + 2), "iteration:", n)

    rel_errors.append((round(relerr, int(-np.log10(relerr)) + 2), n))
    ssq = ssq_new

    if new_imp:
        return f, rel_errors
    else:
        return f, relerr, n

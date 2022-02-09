import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz, norm
from scipy.fftpack import rfft
from pathlib import Path

min_sample_points = 500
max_sample_points = 3550


def load_and_correction(path):
    data = np.loadtxt(path, usecols=(0, 1))
    t = data.T[0]

    end_ind = max_sample_points # len(data.T[1])
    start_ind = min_sample_points

    offset = np.mean(data.T[1][0:1000])
    normalization = 1 / max(data.T[1][start_ind:end_ind] - offset)

    y = (data.T[1][start_ind:end_ind] - offset) * normalization

    return t[start_ind:end_ind], y


def shrinkage_factor(H):
    H = np.array(H, dtype=np.float32)
    H = np.dot(H, H.T)
    return 1 / max(rfft(H[0]))  # since H is circulant -> H.T @ H is -> eig vals with fft

def toeplitz_j(ref):  # convolve(ref, y, mode="wrap") from scipy.ndimage.filters
    c = np.hstack((ref, ref[0]))
    c = toeplitz(c, c[-1::-1])
    return c[0:-1, 0:-1]

def ista_deconvolve(ref, sample):
    """
    sparse deconvolution using an iterative shrinkage algorithm
    :param ref: reference data array
    :param sample: sample data array
    :param tau: iteration step size
    :return: f: sparse deconvolution array,
    relerr: 1-norm(sum abs(f)) of f compared to previous iteration,
    n: number of iterations
    """
    lambda_ = 10
    eps = 1e-10
    max_iteration_count = 2000
    step_scale = 0.3  # tau = tau_scale / norm(a, 2)

    H = toeplitz_j(ref)
    tau = step_scale * shrinkage_factor(H)

    def soft_threshold(v):
        ret = np.zeros(v.shape)

        id_smaller = v <= -lambda_ * tau
        id_larger = v >= lambda_ * tau
        ret[id_smaller] = v[id_smaller] + lambda_ * tau
        ret[id_larger] = v[id_larger] - lambda_ * tau

        return ret

    def calc(H, sample, f):
        return np.dot(H.T, np.dot(H, f) - sample)

    # init variables
    opt_sol = [np.zeros(ref.shape), 1, 0]
    relerr = 1
    n = 0

    f = np.zeros(ref.shape)
    ssq = norm(f, 1)
    big_f = []

    while relerr > eps and n < max_iteration_count:
        n += 1
        pre_calc = calc(H, sample, f)
        f = soft_threshold(f - tau * pre_calc)

        big_f.append(0.5 * norm(sample - H @ f, 2) ** 2 + lambda_ * norm(f, 1))
        print("F(f):", big_f[-1])

        ssq_new = norm(f, 1)
        relerr = abs(1 - ssq / ssq_new)
        if not n % 100:
            print("relerr:", round(relerr, int(-np.log10(relerr)) + 2), "iteration:", n)
        ssq = ssq_new

        opt_sol[0], opt_sol[1], opt_sol[2] = f, relerr, n

    print()
    print("relerr of last iteration: {}".format(relerr), "completed iterations: {}".format(n), "\n")
    print("err. of output solution: {}".format(opt_sol[1]), "at iteration: {}".format(opt_sol[2]))

    return opt_sol[0], opt_sol[1], opt_sol[2]


def deconv_plot(t, y, ref, H, deconvolution, sample_file_path):
    if np.log10(t[1]) <= -12:
        t = t * 10**12

    base_name = str(sample_file_path)
    plot_file_name = base_name + ".png"

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    axs[0].plot(t, ref, label="reference")
    axs[0].plot(t, y, label="sample")
    axs[0].plot(t, H @ deconvolution, label="reconstructed")
    axs[0].set_xlabel("Time (ps)")
    axs[0].set_ylabel("Amplitude (a.u)")
    axs[0].legend()

    axs[1].plot(t, deconvolution)
    axs[1].set_xlabel("Optical delay (ps)")
    axs[1].set_ylabel("Amplitude (a.u)")

    # plt.savefig(plot_file_name, dpi=300)

    plt.show()


if __name__ == '__main__':

    ref_path, sample_path = Path("data/set1/metal_pe_ref.txt"), Path("data/set1/ps-air_420-pe.txt")
    #ref_path, sample_path = Path("data/set2/2018-11-23T17-45-25.074290-ref_buttom.txt"), Path("data/set2/2018-11-23T17-52-20.153015-ps-air-buttom_plate.txt")
    t_ref, ref = load_and_correction(ref_path)
    t_y, y = load_and_correction(sample_path)

    plt.plot(t_ref, ref, label='ref')
    plt.plot(t_y, y, label='sample')
    plt.legend()
    plt.show()

    f, relerr, n = ista_deconvolve(ref, y)

    H = toeplitz_j(ref)
    deconv_plot(t_ref, y, ref, H, f, sample_path)

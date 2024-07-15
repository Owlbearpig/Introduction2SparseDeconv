import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz, norm
from scipy.fftpack import rfft
from pathlib import Path
from implementation import load_and_correction, ista_deconvolve, toeplitz_j
from tttt import ista_deconvolve as ista_og

iterations = 200
lambda_ = 2
step = 1  #0.1


def deconv_plot(t, y, ref, H, deconvolution, sample_file_path, fig_num):
    if np.log10(t[1]) <= -12:
        t = t * 10 ** 12

    base_name = str(sample_file_path)
    plot_file_name = base_name + ".png"

    fig, axs = plt.subplots(2, 1, constrained_layout=True, num=fig_num)

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


def davor():
    dir_ = Path("/home/ftpuser/ftp/Data/Misc/MariaLisa/Baby/Pkt_vor")

    ref_path = dir_ / Path("2022-03-24T20-18-03.394294-Reference-[1]-[0,0,0]-[0,0,0,0]-delta[0mm-0deg]-avg100.txt")
    sam_file = ("2022-03-24T23-16-29.874466-MarienstatueBabyhuefte-[508]-"
                "[59.0,-43.0,-169.42]-[0.98,0.18,-0.09,0.0]-delta[0.009mm-0.0deg]-avg20.txt")
    sam_path = dir_ / Path(sam_file)
    bkg_path = dir_ / Path("2022-03-24T20-11-59.809820-Background-[1]-[0,0,0]-[0,0,0,0]-delta[0mm-0deg]-avg100.txt")

    t_ref, ref = load_and_correction(ref_path, bkg_path)
    t_y, y = load_and_correction(sam_path, bkg_path)
    """
    plt.figure("Davor")
    plt.plot(t_ref, ref, label='ref')
    plt.plot(t_y, y, label='sample')
    plt.legend()
    """
    f, _ = ista_deconvolve(ref, y, new_imp=True, max_iterations=iterations, lambda_=lambda_, step_scale=step)
    # f = ista_og(ref, y, lambda_=lambda_)
    min_pos, max_pos = t_ref[np.argmin(f)], t_ref[np.argmax(f)]
    print("Davor", min_pos, max_pos, min_pos - max_pos)

    H = toeplitz_j(ref)
    deconv_plot(t_ref, y, ref, H, f, sam_path, "Davor_sd")


def danach():
    dir_ = Path("/home/ftpuser/ftp/Data/Misc/MariaLisa/Baby/Pkt_nach")

    ref_path = dir_ / Path("2022-11-08T13-41-37.575984-MetallRef10Avg-[1]-[0,0,0]-[0,0,0,0]-delta[0mm-0deg]-avg10.txt")
    sam_file = ("2022-11-08T12-02-56.415682-Babyhuefte-[415]-[-11.0,-8.0,-26.74]-"
                "[0.98,0.18,-0.01,0.0]-delta[0.018mm-0.0deg]-avg10.txt")
    sam_path = dir_ / Path(sam_file)
    bkg_path = dir_ / Path(
        "2022-11-08T11-02-17.633066-ReferenzLuft100Avg-[1]-[0,0,0]-[0,0,0,0]-delta[0mm-0deg]-avg100.txt")

    t_ref, ref = load_and_correction(ref_path, bkg_path)
    t_y, y = load_and_correction(sam_path, bkg_path)
    """
    plt.figure("Danach")
    plt.plot(t_ref, ref, label='ref')
    plt.plot(t_y, y, label='sample')
    plt.legend()
    """
    f, _ = ista_deconvolve(ref, y, new_imp=True, max_iterations=iterations, lambda_=lambda_, step_scale=step)
    # f = ista_og(ref, y, lambda_=lambda_)
    min_pos, max_pos = t_ref[np.argmin(f)], t_ref[np.argmax(f)]
    print("Danach", min_pos, max_pos, min_pos - max_pos)

    H = toeplitz_j(ref)
    deconv_plot(t_ref, y, ref, H, f, sam_path, "Danach_sd")


if __name__ == '__main__':
    davor()
    danach()
    plt.show()

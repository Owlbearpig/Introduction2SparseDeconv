from implementation import ista_deconvolve
from pathlib import Path
import numpy as np
import multiprocessing

base_dir = Path(r"/home/ftpuser/ftp/Data/Misc/MariaLisa/Baby/CompleteAreas")

skip_before_idx = 0
test_mode = True  # enable(False) or disable(True) saving
start_ind = 0
end_ind = 2000


def eval_area(lam):
    meas_sets = ["before", "after"]
    for meas_set in meas_sets:
        cache_dir = base_dir / "Intermediate Results" / "AreaEval" / meas_set / f"l{lam}"
        cache_dir.mkdir(exist_ok=True, parents=True)

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

        stat_file = cache_dir / "info.txt"
        for i, (t, y, filename) in enumerate(zip(t_axes, y_axes, filenames)):
            if i < skip_before_idx:
                continue

            t_new, y_corrected = pre_process(t, y)
            f, rel_errors, = ista_deconvolve(y_ref_corrected, y_corrected, True, lam, max_iterations=30)

            if not test_mode:
                np.savetxt(str(cache_dir / filename), np.array([t_new, f]).T)
                with open(stat_file, 'a') as file:
                    s = f"{filename} relerr: {rel_errors[-1][0]}, iteration: {rel_errors[-1][1]}"
                    s += f" lambda: {lam}"
                    file.write(s + "\n")
                print("Completed " + s)
            print(i)


if __name__ == '__main__':
    eval_area(12)
    num_workers = multiprocessing.cpu_count()
    lambdas = np.arange(1, 14, 1)

    # split lambdas into parts with num_workers in each
    lambda_chunks = [list(lambdas[i:i + num_workers]) for i in range(0, len(lambdas), num_workers)]

    # doesn't work if each function argument is not iterable...
    for worker_chunk in lambda_chunks:
        listed_lams = [[lam] for lam in worker_chunk]

        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(eval_area, listed_lams)

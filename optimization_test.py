import numpy as np
import numba as nb
from scipy.linalg import toeplitz, norm
import timeit
from numpy.fft import rfft, irfft


np.random.seed(420)

n = 2 ** 11 - 2
ref = np.random.random(n)
sam = np.random.random(n)
f = np.random.random(n)


def toeplitz_j(ref_):  # convolve(ref, y, mode="wrap") from scipy.ndimage.filters
    c = np.hstack((ref_, ref_[0]))
    c = toeplitz(c, c[-1::-1])
    return c[0:-1, 0:-1]


H = toeplitz_j(ref)


def calc(H_, sam_, f_):
    return np.dot(H_.T, np.dot(H_, f_) - sam_)


# @nb.njit(cache=True)
def cir_calc(H_, sam_, f_):
    dot1 = irfft(np.multiply(rfft(ref), rfft(f_)))
    res = irfft(np.multiply(rfft(ref), rfft(dot1 - sam_)))
    return res


def cir_dot(H_, c_):
    return irfft(np.multiply(rfft(H_[:, 0]), rfft(c_)))


def dot(H_, f_):
    return np.dot(H_, f_)


t0 = timeit.default_timer()

for i in range(200):
    if i % 200 == 0:
        print(i)
    calc(H, sam, f)

print("standard matrix mult: ", timeit.default_timer() - t0)

times = []
for i in range(100):
    t0 = timeit.default_timer()

    for i in range(200):
        cir_calc(H, sam, f)
    t1 = timeit.default_timer() - t0
    times.append(t1)

print(times)
print(np.mean(times))
print(calc(H, sam, f))
print(cir_calc(H, sam, f))

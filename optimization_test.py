import numpy as np
from scipy.linalg import toeplitz, norm
import timeit
from numpy.fft import rfft, irfft

np.random.seed(420)

n = 2000
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


def cir_calc(H_, sam_, f_):
    return cir_dot(H_.T, cir_dot(H_, f_) - sam_)


def cir_dot(H_, c_):
    return irfft(np.multiply(rfft(H_[:, 0]), rfft(c_)))


def dot(H_, f_):
    return np.dot(H_, f_)


t0 = timeit.default_timer()

for i in range(1000):
    if i % 200 == 0:
        print(i)
    # dot(H, f)
    # calc(H, sam, f)

print("standard matrix mult: ", timeit.default_timer() - t0)

t0 = timeit.default_timer()

for i in range(200):
    if i % 100 == 0:
        print(i)
    # cir_dot(H, f)
    cir_calc(H, sam, f)

print("fft: ", timeit.default_timer() - t0)

print(calc(H, sam, f))
print(cir_calc(H, sam, f))

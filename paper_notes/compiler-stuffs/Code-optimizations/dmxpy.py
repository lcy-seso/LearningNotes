import time
import numpy as np
np.random.seed(1234)


def dmxpy(n1, y, n2, x, m):
    for j in range(n2):
        for i in range(n1):
            y[i] = y[i] + x[j] * m[i, j]


def dmxpy_op(n1, y, n2, x, m):
    def __unroll_1(jmin):
        y[i] = y[i] + x[jmin] * m[i, jmin]

    def __unroll_2(jmin):
        y[i] = y[i] + x[jmin] * m[i, jmin] + x[jmin + 1] * m[i, jmin + 1]

    def __unroll_4(jmin):
        y[i] = (y[i] + x[jmin] * m[i, jmin] + x[jmin + 1] * m[i, jmin + 1] +
                x[jmin + 2] * m[i, jmin + 2] + x[jmin + 3] * m[i, jmin + 3])

    def __unroll_8(jmin):
        y[i] = (y[i] + x[jmin] * m[i, jmin] + x[jmin + 1] * m[i, jmin + 1] +
                x[jmin + 2] * m[i, jmin + 2] + x[jmin + 3] * m[i, jmin + 3] +
                x[jmin + 4] * m[i, jmin + 4] + x[jmin + 5] * m[i, jmin + 5] +
                x[jmin + 6] * m[i, jmin + 6] + x[jmin + 7] * m[i, jmin + 7])

    remainder = n2 % 16
    width = n2 - remainder
    jmin = width  # The setup loops
    for i in range(n1):
        if remainder == 1:
            __unroll_1(jmin)
        elif remainder == 2:
            __unroll_2(jmin)
        elif remainder == 4:
            __unroll_4(jmin)
        elif remainder == 8:
            __unroll_8(jmin)
        elif remainder == 3:
            __unroll_1(jmin)
            __unroll_2(jmin + 1)
        elif remainder == 5:
            __unroll_1(jmin)
            __unroll_4(jmin + 1)
        elif remainder == 7:
            __unroll_1(jmin)
            __unroll_2(jmin + 1)
            __unroll_4(jmin + 3)
        elif remainder == 9:
            __unroll_1(jmin)
            __unroll_8(jmin + 1)
        elif remainder == 10:
            __unroll_2(jmin)
            __unroll_8(jmin + 2)
        elif remainder == 11:
            __unroll_1(jmin)
            __unroll_2(jmin + 1)
            __unroll_8(jmin + 3)
        elif remainder == 12:
            __unroll_4(jmin)
            __unroll_8(jmin + 4)
        elif remainder == 13:
            __unroll_1(jmin)
            __unroll_4(jmin + 1)
            __unroll_8(jmin + 5)
        elif remainder == 14:
            __unroll_2(jmin)
            __unroll_4(jmin + 2)
            __unroll_8(jmin + 6)
        elif remainder == 15:
            __unroll_1(jmin)
            __unroll_2(jmin + 1)
            __unroll_4(jmin + 3)
            __unroll_8(jmin + 7)

    for jmin in range(0, width, 16):
        for i in range(n1):
            y[i] = (
                y[i] + x[jmin] * m[i, jmin] + x[jmin + 1] * m[i, jmin + 1] +
                x[jmin + 2] * m[i, jmin + 2] + x[jmin + 3] * m[i, jmin + 3] +
                x[jmin + 4] * m[i, jmin + 4] + x[jmin + 5] * m[i, jmin + 5] +
                x[jmin + 6] * m[i, jmin + 6] + x[jmin + 7] * m[i, jmin + 7] +
                x[jmin + 8] * m[i, jmin + 8] + x[jmin + 9] * m[i, jmin + 9] +
                x[jmin + 10] * m[i, jmin + 10] +
                x[jmin + 11] * m[i, jmin + 11] +
                x[jmin + 12] * m[i, jmin + 12] +
                x[jmin + 13] * m[i, jmin + 13] +
                x[jmin + 14] * m[i, jmin + 14] +
                x[jmin + 15] * m[i, jmin + 15])


if __name__ == "__main__":
    n1 = 112
    n2 = 1069

    y = np.asfortranarray(np.random.rand(n1))
    y_copy = y.copy()
    x = np.asfortranarray(np.random.rand(n2))
    # m is stored in column major format.
    m = np.asfortranarray(np.random.rand(n1, n2))
    print(m.flags)

    dmxpy(n1, y, n2, x, m)
    dmxpy_op(n1, y_copy, n2, x, m)
    assert np.allclose(y, y_copy)

    dmxpy(n1, y, n2, x, m)
    for _ in range(10):
        dmxpy(n1, y, n2, x, m)
    start = time.time()
    for _ in range(10):
        dmxpy(n1, y, n2, x, m)
    print("dmxpy time elpased : ", time.time() - start)

    start = time.time()
    dmxpy_op(n1, y_copy, n2, x, m)
    for _ in range(10):
        dmxpy_op(n1, y_copy, n2, x, m)
    start = time.time()
    for _ in range(10):
        dmxpy_op(n1, y_copy, n2, x, m)
    print("dmxpy_op time elpased : ", time.time() - start)

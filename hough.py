import time
import numpy as np
from scipy.sparse import csr_matrix


def hough_lines_2(img):

    h, w = img.shape[0], img.shape[1]

    q = np.ceil( np.sqrt(h*h + w*w) )
    n_rho = int(2 * q - 1)

    #thetas = np.linspace(-90, 89, 180)
    thetas = np.linspace(-5, 5, 16)
    n_theta = len(thetas)

    ts = thetas * np.pi / 180
    cos = np.cos(ts).reshape(-1, 1)
    sin = np.sin(ts).reshape(-1, 1)

    H = np.zeros((n_rho, n_theta), dtype='int')

    r = np.where(img)
    xs, ys = r[0], r[1]

    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)

    t0 = time.time()
    #i = 0

    rhos = xs * cos.T + ys * sin.T
    #sparse = csr_matrix((data, (rhos_idx, thetas_idx)), (nRho, nTheta))

    #for x, y in zip(xs, ys):
    #    i += 1
    #    pass
    #    #print(x, y, time.time())
    #    #for t in thetas:
    #    #    rho = x * np.cos(t) + y * np.sin(t)
    t1 = time.time()

    print('elapsed', t1 - t0)
    return H


def hough_lines_(img):

    iH, iW = img.shape[0], img.shape[1]

    #a0, af, an = (88, 92, 5)
    a0, af, an = (-3, 3, 9)
    #a0, af, an = (-90, 89, 180)
    theta = np.linspace(a0, af, an)
    nTheta = len(theta)

    d = np.sqrt(iH*iH + iW*iW)
    q = int(np.ceil(d))

    #n_rho = int(2 * q - 1)
    #rho = np.linspace(-q, q, nRho)
    #slope = (n_rho - 1) / (q + q)
    n_rho = q
    rho = np.linspace(0, q, n_rho)
    slope = (n_rho - 1) / q

    #h = np.zeros((nRho, nTheta))
    h = csr_matrix((n_rho, nTheta))
    r = np.where(img)
    x, y = r[0], r[1]
    val = img[x, y]

    #take = 50000
    take = 5000

    totK = int(np.ceil(len(val) / take))
    ts = theta * np.pi / 180
    cos = np.cos(ts).reshape(-1, 1)
    sin = np.sin(ts).reshape(-1, 1)

    #totK = 1
    for k in range(totK):

        print(time.time(), "start")

        first = k * take
        last = min(first + take, len(x))

        xs = x[first:last].reshape(-1, 1)
        ys = y[first:last].reshape(-1, 1)

        print(time.time(), "rhos")
        rhos = xs * cos.T + ys * sin.T

        slope = (n_rho - 1) / (rho[-1] - rho[0])

        norm_rho = rhos - rho[0]
        #rho_bin_index = np.round(slope * (norm_rho) + 1)
        rho_bin_index = (slope * (norm_rho) + 1).astype(int)
        print(time.time(), "bin idx")

        num = last-first
        theta_bin_index = np.tile(range(nTheta), (num, 1))

        data = np.ones(num * nTheta).reshape(-1)
        rhos_idx = rho_bin_index.reshape(-1)
        thetas_idx = theta_bin_index.reshape(-1)
        print(time.time(), "sparse")
        sparse = csr_matrix((data, (rhos_idx, thetas_idx)), (n_rho, nTheta))
        #h = h + sparse.toarray()
        print(time.time(), "sum")
        h += sparse


    return h.toarray()



def hough_lines(img):

    iH, iW = img.shape[0], img.shape[1]

    theta = np.linspace(-90, 89, 180)
    nTheta = len(theta)

    d = np.sqrt((iH - 1) ^ 2 + (iW - 1) ^ 2)
    q = np.ceil(d)
    nRho = int(2 * q - 1)
    rho = np.linspace(-q, q, nRho)

    slope = (nRho - 1) / (q + q)
    h = np.zeros((nRho, nTheta))
    r = np.where(img)
    x, y = r[0], r[1]
    val = img[x, y]

    take = 5000

    totK = int(np.ceil(len(val) / take))
    for k in range(1, totK):

        first = (k - 1) * take
        last = min(first + take, len(x))

        x_matrix = np.tile(x[first:last], (1, nTheta))
        y_matrix = np.tile(y[first:last], (1, nTheta))
        val_matrix = np.tile(val[first:last], (1, nTheta))
        tsize = x_matrix.shape[0] # size(x_matrix, 1)
        theta_matrix = np.tile(theta, (tsize, 1)) * np.pi / 180

        rho_matrix = x_matrix * np.cos(theta_matrix) + y_matrix * np.sin(theta_matrix)

        slope = (nRho - 1) / (rho[-1] - rho[0])

        rho_bin_index = round(slope * (rho_matrix - rho[0]) + 1)

        theta_bin_index = np.tile(range(1, nTheta), (tsize, 1))

        sparse = csr_matrix(val_matrix[:], (rho_bin_index[:], theta_bin_index[:]), (nRho, nTheta))
        h = h + np.full(sparse)

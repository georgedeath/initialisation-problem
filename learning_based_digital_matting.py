# Based on:
# Y. Zheng and C. Kambhamettu, “Learning based digital matting,” in
# IEEE International Conference on Computer Vision, September 2009, pp. 889–896.
#
# Python implementation of:
#   http://uk.mathworks.com/matlabcentral/fileexchange/31412-learning-based-digital-matting
#
# Laplacian calculation adapted from http://github.com/MarcoForte/learning-based-matting (db5417a)

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import sparse
from scipy.sparse import linalg as splinalg


def getC(mask, c):
    scribble_mask = mask != 0
    return c * sparse.diags(c * scribble_mask.ravel())


def getAlpha_star(mask):
    alpha_star = np.zeros_like(mask, dtype="float")
    alpha_star[mask > 0] = 1
    alpha_star[mask < 0] = -1
    return alpha_star


def getLapCoefRow(win, numPixInWindow, d, _lambda):
    win = np.concatenate((win, np.ones((win.shape[0], numPixInWindow, 1))), axis=2)

    I = np.tile(np.eye(numPixInWindow), (win.shape[0], 1, 1))
    I[:, -1, -1] = 0

    winTProd = np.einsum("...ij,...kj ->...ik", win, win)
    winTProd_reg_inv = np.linalg.inv(winTProd + I * _lambda)

    F = np.einsum("...ij,...jk->...ik", winTProd, winTProd_reg_inv)
    I_F = np.eye(numPixInWindow) - F

    return np.einsum("...ji,...jk->...ik", I_F, I_F)


def getLap(imdata, mask, winsz, _lambda):
    ih, iw, d = imdata.shape

    # unravel the image to be (npixels, d)
    ravelImg = imdata.reshape(ih * iw, d)

    ravelImg = ravelImg / 255  # normalise image

    numPixInWindow = winsz ** 2
    numPixInWindowSQ = numPixInWindow ** 2
    halfwinsz = winsz // 2

    scribble_mask = mask != 0
    numPix4Training = np.sum(
        1 - scribble_mask[halfwinsz:-halfwinsz, halfwinsz:-halfwinsz]
    )
    numNonzeroValue = numPix4Training * numPixInWindowSQ

    row_inds = np.zeros(numNonzeroValue)
    col_inds = np.zeros(numNonzeroValue)
    vals = np.zeros(numNonzeroValue)

    # create matrix of pixel indices for each window
    A = np.reshape(np.arange(ih * iw), (ih, iw))
    inds_shape = (A.shape[0] - winsz + 1, A.shape[1] - winsz + 1, winsz, winsz)
    inds_strides = (A.strides[0], A.strides[1]) + A.strides
    pixInds = as_strided(A, shape=inds_shape, strides=inds_strides)
    pixInds = np.reshape(
        pixInds, (ih - 2 * halfwinsz, iw - 2 * halfwinsz, numPixInWindow)
    )

    offset = 0

    for i in range(ih - 2 * halfwinsz):
        # select indices of all windows centred on the i'th row
        inds_o = pixInds[i, :]

        # select only those with an unknown label - i.e. 0
        inds = inds_o[~scribble_mask[i + halfwinsz, halfwinsz : iw - halfwinsz]]

        # if we don't have any then skip to the next row
        if not np.any(inds):
            continue

        # calculate the Laplacian coeffs
        win = ravelImg[inds]
        lapcoeff = getLapCoefRow(win, numPixInWindow, d, _lambda)

        # store pixel indices and their corresponding Laplacian value
        step = win.shape[0] * numPixInWindowSQ
        vals[offset : offset + step] = lapcoeff.ravel()
        row_inds[offset : offset + step] = np.repeat(inds, numPixInWindow).ravel()
        col_inds[offset : offset + step] = np.tile(inds, numPixInWindow).ravel()
        offset += step

    return sparse.csr_matrix((vals, (row_inds, col_inds)), shape=(ih * iw, ih * iw))


def solveQuadOpt(L, C, alpha_star):
    lbda = 1e-9  # different lambda to the one used to calc the lap
    # this one regularises the alpha calculation

    D = sparse.eye(L.shape[0])

    alpha = splinalg.spsolve(L + C + D * lbda, C @ alpha_star.ravel())
    alpha = np.reshape(alpha, alpha_star.shape)

    # rescale alpha to ~ [0, 1] if using [-1, 0, 1] as labels
    if np.min(alpha_star) == -1:
        alpha = alpha / 2 + 0.5

    # clip to [0, 1]
    return np.maximum(np.minimum(alpha, 1), 0)


def learningBasedMatting(frame, scribble_mask, winsz=3, c=800, _lambda=1e-07):
    # if the image is a 2d array, add third dim for slicing in other functions
    if len(frame.shape) == 2:
        frame = np.reshape(frame, (*frame.shape, 1))

    C = getC(scribble_mask, c)
    alpha_star = getAlpha_star(scribble_mask)
    L = getLap(frame, scribble_mask, winsz, _lambda)

    return solveQuadOpt(L, C, alpha_star)

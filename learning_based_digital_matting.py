# Based on:
# Y. Zheng and C. Kambhamettu, “Learning based digital matting,” in 
# IEEE International Conference on Computer Vision, September 2009, pp. 889–896.
#
# Python implementation of:
#   http://uk.mathworks.com/matlabcentral/fileexchange/31412-learning-based-digital-matting

import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy import sparse

def getC(mask, c):
    scribble_mask = mask != 0
    mask_area = np.prod(mask.shape)
    C = c * sparse.diags(scribble_mask.ravel(order='F'), offsets=0, 
                         shape=(mask_area, mask_area), dtype='float').tocsr()
    return C

def getAlpha_star(mask):
    alpha_star = np.zeros_like(mask, dtype='float')
    alpha_star.flat[mask.ravel() > 0] = 1
    alpha_star.flat[mask.ravel() < 0] = -1
    return alpha_star

def getLap(imdata, winsz, mask, _lambda):
    # convert to square window if only supplied side
    if type(winsz) == int:
        winsz = np.array([winsz, winsz])
        
    # normalize image data
    imdata = imdata.astype('float') / 255
    
    ih, iw, d = imdata.shape
    pixInds = np.reshape(np.arange(ih*iw), (ih, iw), order="F")

    numPixInWindow = np.prod(winsz)
    numPixInWindowSQ = numPixInWindow**2
    halfwinsz= ((winsz-1)/2).astype('int')
    
    # erode scribble mask by window size
    scribble_mask = mask != 0
    
    struct_ele = np.ones((max(winsz), max(winsz)))
    scribble_mask = binary_erosion(scribble_mask, struct_ele)
    
    # storage for lap values
    numPix4Training = np.sum(1-scribble_mask[halfwinsz[0]+1:-halfwinsz[0], 
                                             halfwinsz[1]+1:-halfwinsz[1]])
    numNonzeroValue = numPix4Training * numPixInWindow**2
    
    row_inds = np.zeros(numNonzeroValue)
    col_inds = np.zeros(numNonzeroValue)
    vals = np.zeros(numNonzeroValue, dtype='float64')
    
    # repeat on each legal pixel
    offset = 0
    
    winData = np.zeros((winsz[0], winsz[1], d))
    win_resized = np.zeros((numPixInWindow, d))
    
    for i in range(halfwinsz[0], ih-halfwinsz[0]):
        for j in range(halfwinsz[1], iw-halfwinsz[1]):

            if scribble_mask[i, j]:
                continue

            # calculate indices of pixels to place in the Laplacian
            win_inds = pixInds[i-halfwinsz[0]:i+halfwinsz[0]+1,
                               j-halfwinsz[1]:j+halfwinsz[1]+1].ravel(order='F')
            row_inds[offset:offset+numPixInWindowSQ] = np.matlib.repmat(win_inds, 1, numPixInWindow)
            col_inds[offset:offset+numPixInWindowSQ] = np.matlib.repmat(win_inds, numPixInWindow, 1).ravel(order='F')

            # calculate the Laplacian on a window
            winData[:] = imdata[i-halfwinsz[0]:i+halfwinsz[0]+1, j-halfwinsz[1]:j+halfwinsz[1]+1, :]
            win_resized[:] = np.reshape(winData, (numPixInWindow, d), order="F")
            vals[offset:offset+numPixInWindowSQ] = compLapCoeff(win_resized, numPixInWindow, d, _lambda).ravel(order='F')

            offset += numPixInWindowSQ

    return sparse.csr_matrix((vals, (row_inds, col_inds)), shape=(ih*iw, ih*iw))
    
def compLapCoeff(win, numPixInWindowSQ, d, _lambda):
    Xi = np.empty((numPixInWindowSQ, d+1))
    Xi[:, :-1] = win
    Xi[:, -1:] = 1

    I = np.eye(numPixInWindowSQ)
    I[-1, -1] = 0
    
    X = Xi @ Xi.T

    # need the transpose here as matrix is transpose of answer in Matlab
    #F = (np.linalg.inv(X+(_lambda*I)) @ X).T  <-- slower
    F = np.linalg.solve(a=(X+(_lambda*I)).T, b=X.T).T
    
    I_F = np.eye(numPixInWindowSQ) - F
    return I_F.T @ I_F

def solveQuadOpt(L, C, alpha_star):
    lbda = 1e-9 # different lambda to the one used to calc the lap
                # this one regularises the alpha calculation

    D = sparse.eye(L.shape[0], L.shape[1]).tocsr()
    
    alpha = sparse.linalg.spsolve(L + C + (D*lbda), C @ alpha_star.ravel(order='F'))
    alpha = np.reshape(alpha, ((alpha_star.shape[0], alpha_star.shape[1])), order='F')
    
    # rescale alpha to ~ [0, 1]
    if np.min(alpha_star) == -1:
        alpha = alpha/2 + 0.5
    
    # clip to [0, 1]
    return np.maximum(np.minimum(alpha, 1), 0)

def learningBasedMatting(frame, scribble_mask, winsz=3, c=800, _lambda=1e-07):
    # if the image is a 2d array, add third dim for slicing in other functions
    if len(frame.shape) == 2:
        frame = np.reshape(frame, (*frame.shape, 1))
    
    L = getLap(frame, winsz, scribble_mask, _lambda)
    alpha = solveQuadOpt(L, getC(scribble_mask, c), getAlpha_star(scribble_mask))
    
    return alpha
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2lab
from skimage.measure import regionprops

from sklearn.svm import OneClassSVM
from vlfeat import vl_dsift
from util import *

def extract_colour_histogram(image, labels, n_bins=8, use_lab=False):    
    ih, iw, _ = image.shape
    n_labels = labels.max()+1
    
    _range = np.array([[0, 256], [0, 256], [0, 256]], dtype='float') # for rgb histograms
    
    if use_lab:
        image = rgb2lab(image)
        _range[:] = [[0,100],[-500*25/29, 500*25/29], [-200*25/29, 200*25/29]]
    
    hist = np.zeros((n_labels, n_bins**3))
    
    mask = np.zeros((ih, iw), dtype='bool')
    
    for i in range(n_labels):
        mask[:] = labels == i
        yy, xx = mask.nonzero()
        
        pixels = image[yy, xx, :]
        
        hist[i, :] = np.histogramdd(sample=pixels, bins=n_bins, range=_range)[0].flat

    return hist
    
def extract_RGB_SIFT_features(image, labels):
    n_sp = np.max(labels)+1
    feat_descs = np.zeros((n_sp, 128*3))
    img_superpixel = np.zeros_like(labels, dtype='int')
    
    # extract SIFT features for each colour channel
    f = np.empty((3, ), dtype='object')
    d = np.empty((3, ), dtype='object')
    for n in range(3):
        f[n], d[n] = vl_dsift(image[..., n], size=1, float_descriptors=True)
    
    r = np.arange(f[0].shape[0]) # indices of all features
    
    # find feature desc nearest to centroid and fill in for each channel
    for i in range(n_sp):
        # get centroid of i'th superpixel
        img_superpixel[:] = labels == i
        c = regionprops(img_superpixel)[0].centroid

        # find nearest sift feature location to the centroid
        D = np.sum((f[0] - c)**2, axis=1)
        d_amin = D.argmin()
        
        # see how many are equally close
        equal_mask = D == D[d_amin]
        n_equal = np.count_nonzero(equal_mask)

        # if no draws, pick closest, else randomly pick from the equally closest
        idx = d_amin if n_equal == 1 else np.random.choice(r[equal_mask])
        
        # fill in the feature vector for each image channel
        for n in range(3):
            # pick out which bit of the feature vector we're in and fill it in
            j, k = n*128, (1+n)*128
            feat_descs[i, j:k] = d[n][idx, :]
    
    return feat_descs
    
def extract_RGB_LBP_features(image, labels, size=5, P=8, R=2):
    n_sp = np.max(labels)+1
    hs = size//2
    img_superpixel = np.zeros_like(labels, dtype='int')
    
    # calculate lbp for entire region
    lbp_img = np.empty((3, ), dtype='object')
    for d in range(3):
        lbp_img[d] = local_binary_pattern(image[..., d], P=P, R=R, method='uniform')
    
    feat_desc_size = P+1
    
    feat_descs = np.zeros((n_sp, feat_desc_size*3))
    
    for i in range(n_sp):
        # get centroid of i'th superpixel
        img_superpixel[:] = labels == i
        cy, cx = [np.rint(x).astype('int') for x in regionprops(img_superpixel)[0].centroid]
        
        # extract lbp values in sizeXsize region centred on the centroid
        x0, y0, x1, y1 = cx-hs, cy-hs, cx+hs+1, cy+hs+1
        
        # clip to boundaries of image
        x0 = 0 if x0 < 0 else x0
        y0 = 0 if y0 < 0 else y0
        x1 = image.shape[1]-1 if x1 > image.shape[1]-2 else x1
        y1 = image.shape[0]-1 if y1 > image.shape[0]-2 else y1

        # fill in the feature vector for each image channel
        for d in range(3):
            j, k = d*feat_desc_size, (1+d)*feat_desc_size
            patch = lbp_img[d][y0:y1, x0:x1].flat
            
            fv = np.histogram(patch, bins=np.arange(0, feat_desc_size+1), range=(0, feat_desc_size+1))[0]
            feat_descs[i, j:k] = fv

    return feat_descs
    
def perform_ocsvm_segmentation(image, bbox, gamma=-19, nu=0.250, crop_ratio=2, 
                               feature_name='RGB', return_crop_region=False):
    """
    Performs segmentation using a One-Class SVM and RGB/LAB/SIFT/LBP features.
    
    Arguments:
        image              = MxNxD numpy array containing the image to be segmented.
        bbox               = array containing bounding box of the form 
                             [x0, y0, x1, y1, x2, y2, x3, y3].
        gamma              = RBF kernel’s length-scale - 2**gamma
        nu                 = upper bound on the assumed number of outliers in 
                             the training data - (0, 1)
        crop_ratio         = factor to multiply the axis-aligned version of the
                             bbox by to denote size of area to crop image to.
        feature_name       = Feature descriptor to use, valid options are:
                             'RGB', 'LAB', 'SIFT', 'LBP'.
        return_crop_region = boolean, if True the function also returns a
                             vector containing the region used for superpixeling.
      
    Output:
        image_mask       = boolean mask containing True for pixels labelled as 
                           belonging to the object, and False otherwise.
        [x0, y0, x1, y1] = start/end points for respective image dimensions used
                           for superpixeling (OPTIONAL)
    """

    bbox = np.array(bbox, dtype='float')
    if bbox.shape != (8,):
        raise ValueError('Bounding box must be of the form [x0, y0, x1, y1, x2, y2, x3, y3].')

    if feature_name not in ['RGB', 'LAB', 'SIFT', 'LBP']:
        raise ValueError('Features must be one of: RGB, LAB, SIFT, or LBP.')
        
    if len(image.shape) != 3:
        raise ValueError('Image must be 3D with last dimension presumed to be RGB.')
        
    Ih, Iw  = image.shape[:2]

    bbox_aa = bbox_to_axis_aligned_bbox(bbox)

    # get coords of region to crop
    c_x0, c_y0, c_x1, c_y1 = get_search_region(bbox_aa, image, crop_ratio)

    # crop image and move bbox to cropped image coords
    c_image = image[c_y0:c_y1, c_x0:c_x1].copy()
    c_bbox = bbox.copy()
    c_bbox[::2] -= c_x0
    c_bbox[1::2] -= c_y0

    # superpixel cropped region
    segments, sp_labels = superpixel_image(c_image, c_bbox)

    # extract features
    if feature_name == 'RGB':
        features = extract_colour_histogram(c_image, segments, use_lab=False)
    elif feature_name == 'LAB':
        features = extract_colour_histogram(c_image, segments, use_lab=True)
    elif feature_name == 'SIFT':
        features = extract_RGB_SIFT_features(c_image, segments)
    else:
        features = extract_RGB_LBP_features(c_image, segments)

    # split into train/test
    f_train = features[~sp_labels, :] # 100% outside bbox
    f_test = features[sp_labels, :]   # overlapping/inside bbox

    # standardise using training data's mean + standard deviation
    mn, std = np.mean(f_train, axis=0), np.std(f_train, axis=0)
    f_train -= mn[np.newaxis, :]
    f_test  -= mn[np.newaxis, :] 
    f_train[:, std != 0] /= std[np.newaxis, std != 0]
    f_test[:, std != 0] /= std[np.newaxis, std != 0]

    # train classifier
    clf = OneClassSVM(kernel='rbf', gamma=2.0**gamma, nu=nu)
    clf.fit(f_train, np.ones(f_train.shape[0]))

    # predict and convert into 1 = inside bbox, 0 = outside labels
    pred = clf.predict(f_test) # predicts 1 outside bbox, -1 otherwise
    pred[pred == -1] = 0 # set so 0 = not outside bbox
    pred = ~np.array(pred, dtype='bool') # cast to boolean and inverse (so True = inside bbox)

    # create mask of area classified as belonging to the object
    mask = np.zeros(c_image.shape[:2], dtype='bool')

    for i, sp_no in enumerate(np.arange(len(sp_labels))[sp_labels == True]):
        if pred[i]:
            mask |= (segments == sp_no)

    image_mask = np.zeros(image.shape[:2], dtype='bool')
    image_mask[c_y0:c_y1, c_x0:c_x1] = mask

    if return_crop_region:
        return image_mask, [c_x0, c_y0, c_x1, c_y1]
    
    return image_mask
    
if __name__ == "__main__":
    from scipy.ndimage import imread
    import matplotlib.pyplot as plt
    
    # apply segmentation to gymnastics4 image (from VOT2016 data set)
    image = imread('images/gymnastics4_00000233.jpg')
    bbox = np.array([376.33, 262.1, 461.85, 205.38,
                     500.77, 264.05, 415.25, 320.77])
                     
    mask, [x0, y0, x1, y1] = perform_ocsvm_segmentation(image, bbox, feature_name='LAB', return_crop_region=True)
    
    # remove pixels from image that are labelled as background
    image_masked = image.copy()
    for d in range(3):
        image_masked[..., d].flat[~mask.ravel()] = 255
    
    # display original image, segmentation, and segmented image
    images = [image[y0:y1, x0:x1, :], mask[y0:y1, x0:x1], image_masked[y0:y1, x0:x1]]
    titles = ['Original image', 'Segmentation', 'Segmented image']

    bbox_pts = np.concatenate((np.reshape(bbox, (-1, 2)), bbox[:2][np.newaxis, :]))
    bbox_pts -= [x0, y0]
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for image, title, a in zip(images, titles, ax.flat):
        a.imshow(image)
        for i in range(4):
            a.plot(bbox_pts[i:i+2, 0], bbox_pts[i:i+2, 1], 'c-', lw=2)
        a.set_title(title)
        a.axis('off')
    plt.show()


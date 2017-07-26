import numpy as np
from learning_based_digital_matting import learningBasedMatting
from skimage.draw import polygon
from util import get_search_region, bbox_to_axis_aligned_bbox

def expand_region(x, y, rho, ih, iw, winsz):
    """
    Expands/contracts bbox's area by rho, i.e. new area = rho * old area,
    while constraining to lie within image boundary, padded inwards by
    alpha matting window size.
    """
    # centroid
    xc, yc = x.mean(), y.mean()

    # linearly scale dims by sqrt(rho)
    r = np.sqrt(rho)
    xx = r*(x - xc) + xc
    yy = r*(y - yc) + yc
    
    # clip to image boundary (inner padding by alpha matting window size)
    xx = np.maximum(np.minimum(xx, iw-1-winsz), winsz)
    yy = np.maximum(np.minimum(yy, ih-1-winsz), winsz)
    
    return xx, yy

def create_scribble_mask(x, y, frame, pct_area_shrink=0.1, pct_area_grow=0.1, winsz=3):
    """
    Arguments:
        x               = x-coordinates of bbox
        y               = y-coordinates of bbox
        frame           = image to create scribble mask for
        pct_area_shrink = fraction of bbox area to shrink by: 
                          new area = old area * (1-pct_area_shrink)
        pct_area_grow   = fraction of bbox area to expand by: 
                          new area = old area * pct_area_grow
        winsz           = window size of the alpha matting algorithm
      
    Output:
      scribble_mask = mask for pixels. Contains 1 for definite foreground, 
                      -1 for definite background, and 0 for unknown pixels.
    """
    ih, iw = frame.shape[:2]
    
    # expand bbox by 'pct_area_grow' area
    xe, ye = expand_region(x, y, pct_area_grow, ih, iw, winsz)

    # shrink bbox by 'pct_area_shrink' area
    xs, ys = expand_region(x, y, 1-pct_area_shrink, ih, iw, winsz)

    # pixel indices (row, column) for expanded/contracted bboxes
    re, ce = polygon(xe, ye) # pixels for expanded polygon
    rs, cs = polygon(xs, ys) # pixels for contracted polygon

    # mask for outside of expanded bbox
    expanded_mask = np.zeros((ih, iw), dtype='bool')
    expanded_mask[ce, re] = True # mark inside expanded bbox
    expanded_mask = ~expanded_mask # invert to get outside

    # final scribble mask with -1 for outside expanded bbox, 1 inside, 0 otherwise
    scribble_mask = np.zeros((ih, iw), dtype='int')
    scribble_mask.flat[expanded_mask.ravel()] = -1
    scribble_mask[cs, rs] = 1

    return scribble_mask

def perform_alpha_matting(image, bbox, rho_plus=1.2, rho_minus=0.8, tau=0.84, 
                          lbda=1e-2, crop_ratio=2, return_crop_region=False):
    """
    Performs alpha matting segmentation.
    
    Arguments:
        image              = MxNxD numpy array containing the image to be segmented.
        bbox               = array containing bounding box of the form 
                             [x0, y0, x1, y1, x2, y2, x3, y3].
        rho_plus           = factor to expand bbox by to label area outside this
                             as belonging to the background.
        rho_minus          = factor to contract bbox by to label area inside this
                             as belonging to the object.
        tau                = fraction of area of the original bounding box we 
                             wish the alpha matte to occupy.
        lbda               = ridge regression regularisation term.
        crop_ratio         = factor to multiply the axis-aligned version of the
                             bbox by to denote size of area to crop image to.
        return_crop_region = boolean, if True the function also returns a
                             vector containing the region used for segmentation.
                     
    Output:
        mask             = boolean mask containing True for pixels labelled as 
                           belonging to the object, and False otherwise.
        [x0, y0, x1, y1] = start/end points for respective image dimensions used
                           for superpixeling (OPTIONAL)
    """
    bbox = np.array(bbox, dtype='float')
    if bbox.shape != (8,):
        raise ValueError('Bounding box must be of the form [x0, y0, x1, y1, x2, y2, x3, y3].')

    Ih, Iw  = image.shape[:2]

    bbox_aa = bbox_to_axis_aligned_bbox(bbox)

    # get coords of region to crop
    c_x0, c_y0, c_x1, c_y1 = get_search_region(bbox_aa, image, crop_ratio)

    # crop image and move bbox to cropped image coords
    c_image = image[c_y0:c_y1, c_x0:c_x1, :].copy()
    c_bbox = bbox.copy()
    c_bbox[::2] -= c_x0
    c_bbox[1::2] -= c_y0

    # create scribble mask of the cropped region
    scribble_mask = create_scribble_mask(x=c_bbox[::2], y=c_bbox[1::2], frame=c_image, 
                                         pct_area_shrink=rho_minus, 
                                         pct_area_grow=rho_plus, winsz=3)

    # perform the alpha matting
    alpha = learningBasedMatting(frame=c_image, scribble_mask=scribble_mask, 
                                 winsz=3, _lambda=lbda)

    # threshold
    bbox_area = polygon(bbox[::2], bbox[1::2])[0].size # count num pixels in polygon of bbox
    for t in np.arange(1, 0, -0.001):
        if (np.count_nonzero(alpha > t) / bbox_area) >= tau:
            break

    mask = np.zeros((Ih, Iw), dtype='bool')
    mask[c_y0:c_y1, c_x0:c_x1] = alpha > t
    
    if return_crop_region:
        return mask, [c_x0, c_y0, c_x1, c_y1]
    
    return mask
    
if __name__ == "__main__":
    from scipy.ndimage import imread
    import matplotlib.pyplot as plt
    
    # apply segmentation bag image (from VOT2016 data set)
    image = imread('images/bag_00000001.jpg')
    bbox = np.array([334.02, 128.36, 438.19, 188.78,
                     396.39, 260.83, 292.23, 200.41])
                     
    mask, [x0, y0, x1, y1] = perform_alpha_matting(image, bbox, return_crop_region=True)
    
    # remove pixels from image that are labelled as background
    image_masked = image.copy()
    for d in range(3):
        image_masked[..., d].flat[~mask.ravel()] = 255

    # display original image, segmentation, and segmented image
    images = [image[y0:y1, x0:x1, :], mask[y0:y1, x0:x1], image_masked[y0:y1, x0:x1]]
    titles = ['Original image (cropped)', 'Segmentation', 'Segmented image']
    
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

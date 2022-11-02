import numpy as np

from util import bbox_to_axis_aligned_bbox, get_search_region, superpixel_image


def sbbm(frame, segments, labels, delta=0.5, eta=0.8, R=20):
    # frame = (H, W, D) where D = features, typically RGB
    # segments = superpixel label mask
    # labels = background (False) / unknown (True) label for the superpixel
    #          corresponding to the label's index.

    # delta = proportion of the average superpixel size to sample (0, 1]
    # eta = match threshold - (0, 1]

    # number of samples = delta * avg number of  pixels per superpixel
    S = np.rint(segments.size / labels.size * delta).astype("int")

    # sample (with replacement) each superpixel S times
    samples = np.zeros((sum(~labels), S, frame.shape[2]))
    sp_mask = np.zeros(frame.shape[:2], dtype="bool")
    superpixel_inds = np.arange(labels.size)

    for i, sp_no in enumerate(superpixel_inds[~labels]):
        # i = sample index, sp_no = superpixel number
        sp_mask[:, :] = segments == sp_no
        yy, xx = sp_mask.nonzero()

        inds = np.random.randint(low=0, high=len(yy), size=S)
        samples[i, :, :] = frame[yy[inds], xx[inds], :]

    Rsqr = R * R

    pred = np.ones(sum(labels), dtype="bool")  # predictions for unknown sps

    for i, sp_no in enumerate(superpixel_inds[labels]):
        sp_mask[:, :] = segments == sp_no
        yy, xx = sp_mask.nonzero()
        pixel_values = frame[yy, xx, :]  # shape=(npix, D)

        # calculate squared distance between samples and pixel values
        Z = (
            samples[:, :, np.newaxis, :] - pixel_values[np.newaxis, np.newaxis, :]
        )  # shape(n_training, N, npix, D)
        Z *= Z
        Z = np.sum(Z, axis=3)  # shape=(n_training, S, npix)

        # count number of pixels that match each sample
        Z = np.sum(Z < Rsqr, axis=(1, 2))  # shape = n_training

        # normalise to the range [0, 1]
        samples_matched_pct = Z / (len(yy) * S)  # shape = n_training - [0, 1]

        # if any match the background, predict it as such
        if np.any(samples_matched_pct > eta):
            pred[i] = False

    return pred


def perform_SBBM_segmentation(
    image, bbox, delta=0.5, eta=0.5, crop_ratio=2, return_crop_region=False
):
    """
    Performs segmentation using a One-Class SVM and RGB/LAB/SIFT/LBP features.

    Arguments:
        image             = MxNxD numpy array containing the image to be segmented.
        bbox              = array containing bounding box of the form
                            [x0, y0, x1, y1, x2, y2, x3, y3].
        delta             = proportion of the average superpixel size to sample (0, 1]
        eta               = match threshold - (0, 1]
        crop_ratio        = factor to multiply the axis-aligned version of the
                             bbox by to denote size of area to crop image to.
        return_crop_region = boolean, if True the function also returns a
                             vector containing the region used for superpixeling.
    Output:
        image_mask       = boolean mask containing True for pixels labelled as belonging
                           to the object, and False otherwise.
        [x0, y0, x1, y1] = start/end points for respective image dimensions used
                           for superpixeling (OPTIONAL)
    """

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

    # classify with SBBM
    pred = sbbm(c_image, segments, sp_labels, delta=delta, eta=eta)

    # create mask of area classified as belonging to the object
    mask = np.zeros(c_image.shape[:2], dtype="bool")

    for i, sp_no in enumerate(np.arange(len(sp_labels))[sp_labels == True]):
        if pred[i]:
            mask |= segments == sp_no

    image_mask = np.zeros(image.shape[:2], dtype="bool")
    image_mask[c_y0:c_y1, c_x0:c_x1] = mask

    if return_crop_region:
        return image_mask, [c_x0, c_y0, c_x1, c_y1]

    return image_mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # apply segmentation to gymnastics4 image (from VOT2016 data set)
    image = plt.imread("images/crossing_00000066.jpg")
    bbox = np.array([495.64, 261.45, 543.01, 264.25, 532.31, 445.24, 484.94, 442.44])

    mask, [x0, y0, x1, y1] = perform_SBBM_segmentation(
        image, bbox, return_crop_region=True
    )

    # remove pixels from image that are labelled as background
    image_masked = image.copy()
    for d in range(3):
        image_masked[..., d].flat[~mask.ravel()] = 255

    # display original image, segmentation, and segmented image
    images = [image[y0:y1, x0:x1, :], mask[y0:y1, x0:x1], image_masked[y0:y1, x0:x1]]
    titles = ["Original image (cropped)", "Segmentation", "Segmented image"]

    bbox_pts = np.concatenate((np.reshape(bbox, (-1, 2)), bbox[:2][np.newaxis, :]))
    bbox_pts -= [x0, y0]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for image, title, a in zip(images, titles, ax):
        a.imshow(image)
        for i in range(4):
            a.plot(bbox_pts[i : i + 2, 0], bbox_pts[i : i + 2, 1], "c-", lw=2)
        a.set_title(title)
        a.axis("off")
    plt.show()

import numpy as np
from skimage.draw import polygon
from skimage.segmentation import slic


def get_search_region(bbox, frame, ratio):
    """
    Calculates coordinates of ratio*width/height of axis-aligned bbox, centred
    on the original bbox, constrained by the original size of the image.

    Arguments:
      bbox  = axis-aligned bbox of the form [x0, y0, width, height]
      frame = MxNxD Image to constrain bbox by
      ratio = ratio at which to change bbox dimensions by

    Output:
      x0, y0, x1, y1 = Coordinates of expanded axis-aligned bbox
    """
    x0, y0, w, h = bbox
    ih, iw = frame.shape[:2]

    ww, hh = ratio * w, ratio * h

    # expand bbox by ratio
    x1 = np.min([iw - 1, x0 + w / 2 + ww / 2])
    y1 = np.min([ih - 1, y0 + h / 2 + hh / 2])
    x0 = np.max([0, x0 + w / 2 - ww / 2])
    y0 = np.max([0, y0 + h / 2 - hh / 2])

    x0, y0, x1, y1 = np.rint(np.array([x0, y0, x1, y1])).astype("int")

    return x0, y0, x1, y1


def bbox_to_axis_aligned_bbox(bbox):
    """
    Converts bbox of the form [x0, y0, x1, y1, x2, y2, x3, y3] to
    axis-aligned version, i.e.: [x0, y0, width height]
    """
    bbox_reshaped = bbox.reshape([4, 2])
    bbox_aa = np.array([bbox_reshaped[:, 0].min(), bbox_reshaped[:, 1].min()])

    return np.concatenate((bbox_aa, np.max(bbox_reshaped, axis=0) - bbox_aa))


def superpixel_image(image, bbox, Nsp_min=100, Nsp_max=500, Nsp_npx=50):
    """
    Superpixels an image using slic0 and labels them 1 if they are 100%
    inside the bounding box, 0 otherwise.

    Arguments:
        image = MxNxD
        bbox =
        Nsp_npx = number of pixels per superpixel (on avg) we're aiming for
        Nsp_min = (approx) min number of superpixels in cropped region
        Nsp_max = (approx) max number of superpixels in cropped region

    Output:
        segments = MxN label image where each pixel's value represents the
                   superpixel it belongs to.
        sp_label = boolean vector containing segments.max()+1 entries
                   corresponding to the label of the superpixel, with 1
                   indicating 100% inside the bounding box, 0 otherwise.
    """
    bbox_aa = bbox_to_axis_aligned_bbox(bbox)

    # calculate number of superpixels to aim for
    n_sp = np.rint(bbox_aa[2] * bbox_aa[3] / Nsp_npx).astype("int")
    n_sp = np.max([Nsp_min, np.min([Nsp_max, n_sp])])

    # segment the image using SLIC0
    segments = slic(image, n_segments=n_sp, slic_zero=True, enforce_connectivity=True)
    segments = segments - np.min(segments)
    n_sp = segments.max() + 1  # actual number of superpixels

    # create mask for outside of bounding box
    bbox_mask = np.zeros(image.shape[:2], dtype="bool")
    x, y = polygon(bbox[::2], bbox[1::2])
    bbox_mask[y, x] = True

    # label superpixels - 0 = 100% outside bbox, 1 = some overlap with bbox
    sp_label = np.zeros((n_sp), dtype="bool")

    for n in range(n_sp):
        # label n'th sp as inside bbox if any of its pixels overlap the bbox
        if np.any((segments == n) & bbox_mask):
            sp_label[n] = True

    return segments, sp_label

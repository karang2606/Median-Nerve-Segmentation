import warnings

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import find_contours

def hausdorff_distance_mask(image0, image1, method = 'standard'):
    """Calculate the Hausdorff distance between the contours of two segmentation masks.
    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a pixel from a segmented object. Both arrays must have the same shape.
    method : {'standard', 'modified'}, optional, default = 'standard'
        The method to use for calculating the Hausdorff distance.
        ``standard`` is the standard Hausdorff distance, while ``modified``
        is the modified Hausdorff distance.
    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of the segmentation mask contours in
        ``image0`` and ``image1``, using the Euclidean distance.
    Notes
    -----
    The Hausdorff distance [1]_ is the maximum distance between any point on the 
    contour of ``image0`` and its nearest point on the contour of ``image1``, and 
    vice-versa.
    The Modified Hausdorff Distance (MHD) has been shown to perform better
    than the directed Hausdorff Distance (HD) in the following work by
    Dubuisson et al. [2]_. The function calculates forward and backward
    mean distances and returns the largest of the two.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155
    Examples
    --------
    >>> ground_truth = np.zeros((100, 100), dtype=bool)
    >>> predicted = ground_truth.copy()
    >>> ground_truth[30:71, 30:71] = disk(20)
    >>> predicted[25:65, 40:70] = True
    >>> hausdorff_distance_mask(ground_truth, predicted)
    11.40175425099138
    """
    
    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')
    
    a_points = find_contours(image0>0)
    b_points = find_contours(image1>0)
    
    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    a_points = np.concatenate(a_points)
    b_points = np.concatenate(b_points)
    
    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )

    if method == 'standard':  # standard Hausdorff distance
        return max(max(fwd), max(bwd))
    elif method == 'modified':  # modified Hausdorff distance
        return max(np.mean(fwd), np.mean(bwd))
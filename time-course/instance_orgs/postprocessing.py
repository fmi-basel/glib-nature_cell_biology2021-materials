from typing import Tuple, Iterator
import heapq

import numpy as np
import numba

from sklearn.neighbors import KDTree
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label as connected_components
from scipy.ndimage import minimum_position
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import regionprops


@numba.njit()
def _accumulate_votes(votes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    '''accumulate votes in 2D.
    '''
    accumulated = np.zeros(shape)

    for xy in votes:
        x, y = xy
        x = round(x)
        y = round(y)
        accumulated[x, y] += 1
    return accumulated


def estimate_embedding_centers(embeddings: np.ndarray,
                               fg_mask: np.ndarray,
                               min_votes: int = 9,
                               sigma: float = 1.,
                               nonmax_size: int = 9) -> np.ndarray:
    '''estimate centers within embeddings that can then be used to assign
    pixels to each instance.
    '''
    # scale embeddings to vote space
    lower, upper = np.percentile(embeddings, (0, 100), axis=(0, 1))
    offset = lower
    scale = np.asarray(embeddings.shape[:-1]) / (upper - lower)
    embeddings = (embeddings - offset) * scale

    accumulated = _accumulate_votes(embeddings[fg_mask], embeddings.shape[:-1])

    accumulated = gaussian_filter(accumulated, sigma=sigma)
    maxima_mask = maximum_filter(accumulated, size=nonmax_size) == accumulated
    maxima_mask = np.logical_and(accumulated >= min_votes, maxima_mask)
    maxima_labels, num_components = connected_components(maxima_mask)

    centers = center_of_mass(np.ones_like(maxima_labels),
                             labels=maxima_labels,
                             index=range(1, num_components + 1))

    # handle the case where no centers were found.
    if not centers:
        return []

    # scale back
    centers = centers / scale + offset
    return centers


def get_seeds(centers, embeddings, mask) -> np.ndarray:
    '''determine the location of the closest point in embeddings
    (within the mask) to each center.
    '''
    # We use KDTree to find the closest center for each
    # foreground location, then we search for the minimum within
    # this partition.
    tree = KDTree(centers, leaf_size=1)
    dist, ind = tree.query(embeddings[mask], k=1)
    cond_dist = np.zeros(mask.shape)
    cond_dist[mask] = dist.squeeze()
    regions = np.zeros(mask.shape)
    regions[mask] = ind.squeeze() + 1
    return minimum_position(cond_dist,
                            labels=regions,
                            index=list(range(1,
                                             len(centers) + 1)))


@numba.njit()
def _neighbours(x, y, lbl, fg_mask, labels) -> Iterator[Tuple[int, int]]:
    '''iterates over 4-connected neighbours.
    '''
    for xx, yy in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
        if 0 <= xx < labels.shape[0] and \
           0 <= yy < labels.shape[1] and \
           fg_mask[xx, yy] and \
           labels[xx, yy] != lbl:
            yield xx, yy


@numba.njit()
def _grow(centers, seeds, embeddings,
          fg_mask) -> Tuple[np.ndarray, np.ndarray]:
    '''numba-compiled seed-growing logic for grow_from_centers.

    Returns: labels, distances
    '''
    distances = np.ones(fg_mask.shape) * np.inf
    labels = np.zeros(fg_mask.shape, dtype=np.int32) - 1

    queue = [
        (np.linalg.norm(embeddings[idx[0], idx[1]] - centers[label]), idx,
         label)  # Tuple of (distance, (xy_coord), centroid_label)
        for label, idx in enumerate(seeds)
    ]
    heapq.heapify(queue)

    while queue:
        dist, (x, y), lbl = heapq.heappop(queue)

        if distances[x, y] <= dist:
            continue

        labels[x, y] = lbl
        distances[x, y] = dist

        for xx, yy in _neighbours(x, y, lbl, fg_mask, labels):
            dist = np.linalg.norm(embeddings[xx, yy] - centers[lbl])
            if distances[xx, yy] > dist:
                heapq.heappush(queue, (dist, (xx, yy), lbl))
    return labels + 1, distances


def grow_from_centers(centers: np.ndarray, embeddings: np.ndarray,
                      fg_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''determines the best seed for each center and then creates a
    segmentation by expanding from the points closest to the centers
    in a closest-point-first manner. This ensures that all segments
    are connected.

    Returns: labels, distances
    '''
    seeds = get_seeds(centers, embeddings, fg_mask)
    seeds = numba.typed.List(seeds)  # needed for type inference in _grow
    return _grow(centers, seeds, embeddings, fg_mask)


def embeddings_to_segmentation(fg_mask: np.ndarray, embeddings: np.ndarray,
                               bandwidth: float,
                               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    '''generate an instance segmentation from the predicted fg_mask and
    embeddings. This uses a seeded growth algorithm to prevent disjoint
    objects.
    
    The resulting segmentation may contain objects that have holes. To
    fill those, see fill_holes_in_objects.

    '''

    def _empty_segmentation():
        return np.zeros_like(fg_mask), np.zeros_like(fg_mask, dtype='float32')

    if not np.any(fg_mask):
        return _empty_segmentation()

    centers = estimate_embedding_centers(embeddings, fg_mask=fg_mask, **kwargs)
    if len(centers) == 0:
        return _empty_segmentation()

    segm, dist = grow_from_centers(centers, embeddings, fg_mask)

    # turn distances into pseudo-probabilities:
    probs = np.exp(-(dist.squeeze() / (2 * bandwidth))**2)
    return segm, probs


def fill_holes_in_objects(segmentation: np.ndarray) -> np.ndarray:
    '''fills holes within segmentated objects. Should an object contain
    another, then an overwrite may occur depending on the label order.
    '''
    # guard for empty segmentation -- throws an ugly type error otherwise.
    if np.all(segmentation == 0):
        return segmentation
    for region in regionprops(segmentation):
        np.putmask(segmentation[region.slice], region.filled_image,
                   region.filled_image * region.label)
    return segmentation

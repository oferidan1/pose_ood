import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


def match_descriptors(descriptors1, descriptors2, k=2):
    """
    Compute matches between descriptors using FLANN
    :param descriptors1: N x descriptor_dim
    :param descriptors2: M x descriptor_dim
    :return: list of matches
    """
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    des1 = np.float32(descriptors1)
    des2 = np.float32(descriptors2)
    matches = flann.knnMatch(des1, des2, k=k)
    return matches

def bidirectional_matches(d1, d2, k=1):
    """
    Compute matches between descriptors using tensors' correlation (PyTorch)
    :param d1: (MxK) matrix of M queries (each row is a entry with K elements)
    :param d2: (NxK) matrix of N targets (each row is a entry with K elements)
    :param k: (int) number of neighbors to extract
    :return: list of matches
    """
    _, d1_d2_matches = match_descriptors_torch(d1, d2, k=k)
    _, d2_d1_matches = match_descriptors_torch(d2, d1, k=k)

    i = torch.arange(len(d2_d1_matches), device=d1.device)
    d2_matches = np.where((d1_d2_matches[d2_d1_matches] == i).cpu().numpy())[0]
    d1_matches = d2_d1_matches[d2_matches].cpu().numpy()
    matches = np.stack((d1_matches, d2_matches), axis=1)
    return matches


def bidirectional_matches_torch(d1, d2, k=1):
    """
    Compute matches between descriptors using tensors' correlation (PyTorch)
    :param d1: (MxK) matrix of M queries (each row is a entry with K elements)
    :param d2: (NxK) matrix of N targets (each row is a entry with K elements)
    :param k: (int) number of neighbors to extract
    :return: list of matches
    """
    _, d1_d2_matches = match_descriptors_torch(d1, d2, k=k)
    _, d2_d1_matches = match_descriptors_torch(d2, d1, k=k)

    i = torch.arange(len(d2_d1_matches), device=d1.device)
    d2_matches = torch.where(d1_d2_matches[d2_d1_matches] == i)[0]
    d1_matches = d2_d1_matches[d2_matches]
    matches = torch.stack((d1_matches, d2_matches), axis=1)
    return matches


def match_descriptors_torch(d1, d2, k=1):
    """
    Finds the closest rows in d2 in respect to rows in d1
    :param d1: (MxK) matrix of M queries (each row is a entry with K elements)
    :param d2: (NxK) matrix of N targets (each row is a entry with K elements)
    :param k: (int) number of neighbors to extract
    :return: the distances and indices of d1 to d2 matches
    """
    x_nr = torch.sum(d1 ** 2, dim=1) / 2
    q_nr = torch.sum(d2 ** 2, dim=1) / 2
    q_x = torch.matmul(d1, d2.t())

    sim = q_nr + (x_nr - q_x.t()).t()
    if k == 1:
        dists, indices = torch.min(sim, dim=1)
    else:
        dists, indices = torch.sort(sim, dim=1)[:, :k]

    return dists, indices


def match_descriptors_bt(descriptors1, descriptors2, norm_type=2, cross_check=False):
    """
    Compute matches between descriptors using brute-force search
    :param descriptors1: N x desccriptor_size matrix with descriptors
    :param descriptors2: N x desccriptor_size matrix with descriptors
    :param norm_type: (int) sets the distance type using for brute-force NN search
    :param cross_check: (bool) indicates whether to return on;y bi-directional matches
    :return:
    """
    if norm_type == 1:
        norm_type = cv2.NORM_L1
    elif norm_type == 2:
        norm_type = cv2.NORM_L2
    elif norm_type == 3:
        norm_type = cv2.NORM_HAMMING
    else:
        norm_type = cv2.NORM_L2

    bf = cv2.BFMatcher(norm_type, cross_check)
    des1 = np.float32(descriptors1)
    des2 = np.float32(descriptors2)
    matches = bf.match(des1, des2)
    return matches


def lowe_ratio_test(matches, threshold):
    good_matches = []
    for i, knn_match_pair in enumerate(matches):
        neighbor1, neighbor2 = knn_match_pair
        if neighbor1.distance < threshold * neighbor2.distance:
            good_matches.append(neighbor1)
    return good_matches
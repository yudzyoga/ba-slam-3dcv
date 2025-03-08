#!/usr/bin/env python3
import torch

"""
    This code was taken from my 1st assignment Task 2a, which is about brute force feature matching 
    with Lowe1, Lowe2, and FwdBwd consistency check.
"""

def match(descriptors1, descriptors2, device, dist="norm2", threshold=0, ratio=0.5):
    """
    Brute-force descriptor match with Lowe tests and cross consistency check.


    Inputs:
    - descriptors1: tensor(N, feature_size),the descriptors of a keypoint
    - descriptors2: tensor(N, feature_size),the descriptors of a keypoint
    - device: where a torch.Tensor is or will be allocated
    - dist: distance metrics, hamming distance for measuring binary descriptor, and norm-2 distance for others
    - threshold: threshold for first Lowe test
    - ratio: ratio for second Lowe test

    Returns:
    - matches: tensor(M, 2), indices of corresponding matches in first and second set of descriptors,
      where matches[:, 0] denote the indices in the first and
      matches[:, 1] the indices in the second set of descriptors.
    """

    # Exponent for norm
    if dist == "hamming":
        p = 0
    else:
        p = 2.0

    ######################################################################################################
    # TODO Q1: Find the indices of corresponding matches                                                 #
    # See slide 48 of lecture 2 part A                                                                   #
    # Use cross-consistency checking and first and second Lowe test                                      #
    ######################################################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Compute distances
    # NOTE: you may use torch.cdist with p=p
    """
        Testing procedures:
        - First Lowe test
            The first lowe test only uses a threshold for its shortest distance
            If it exceed the given threshold, it skips the step
        - Second Lowe test
            The second test uses the second shortest distance given the ratio
            if it exceed, it skips the step
        - Forward backward consistency test
            The method to find the best index match between two different descriptors
    """
    verbose = False # to switch on logs
    
    # generate distance matrix, each entry in descriptors1 will have each
    # L2 distances with each of descriptors2 entries.
    distance_matrix = torch.cdist(descriptors1, descriptors2, p).to(device)
    matches = torch.zeros(0, 2).to(device)
    distances = torch.zeros(0).to(device)
    for idx_i, distance_matrix_i in enumerate(distance_matrix):
        if verbose:
            print(f"\n--- Start matching index: {idx_i} ---")

        # for the specific descriptors1 (i), sort each distances
        sorted_distance_i = torch.sort(distance_matrix_i)

        # find 1st and 2nd lowest value distances and the index of the lowest
        dist_i_top1, dist_i_top2 = sorted_distance_i.values[:2]
        idx_j_top1 = sorted_distance_i.indices[0]

        # Perform the first and the second lowe test
        # first test
        if not dist_i_top1 <= threshold:
            if verbose:
                print(f"   Lowe1: Threshold not pass ({dist_i_top1} < {threshold}). skipping")
            continue
        # second test
        if not dist_i_top1 <= (dist_i_top2 * ratio):
            if verbose:
                print(f"   Lowe2: Limit not pass ({dist_i_top1} < {dist_i_top2 * ratio}). skipping")
            continue    

        # Forward backward consistency check
        # take the i-th element and find its minimum
        sorted_distance_j = torch.sort(distance_matrix[:, idx_j_top1])
        idx_i_top1 = sorted_distance_j.indices[0]

        if not (idx_i_top1 == idx_i):
            if verbose:
                print(f"   FwdBwdConsistency: Not consistent (idx_i {idx_i} != idx_i_converted {idx_i_top1}). skipping")
            continue

        if verbose:
            print(f"   All test passed. Match (i, j, dist) = ({idx_i_top1}, {idx_j_top1}, {dist_i_top1})")
        # append the matches and distances for further calculation
        matches = torch.cat((matches, torch.tensor([[idx_i_top1, idx_j_top1]]).to(device)), 0)
        distances = torch.cat((distances, torch.tensor([dist_i_top1]).reshape(-1, 1).to(device)), 0)
    
    # Sort matches using distances from best to worst
    distance_sorted = distances.reshape(-1).sort()
    matches = matches[distance_sorted.indices].type(torch.int)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return matches


if __name__ == "__main__":
    # test your match function under here by using provided image, keypoints, and descriptors
    import numpy as np
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img1 = cv2.imread("../data/Chess.png")
    color1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)

    img2 = cv2.imread("../data/ChessRotated.png")
    color2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(color2, cv2.COLOR_RGB2GRAY)

    keypoints1 = np.loadtxt("./keypoints1.in")
    keypoints2 = np.loadtxt("./keypoints2.in")
    keypoints1 = torch.tensor(keypoints1, device=device)
    keypoints2 = torch.tensor(keypoints2, device=device)

    descriptors1 = np.loadtxt("./descriptors1.in")
    descriptors2 = np.loadtxt("./descriptors2.in")
    descriptors1 = torch.tensor(descriptors1, device=device)
    descriptors2 = torch.tensor(descriptors2, device=device)

    matches = match(
        descriptors1=descriptors1,
        descriptors2=descriptors2,
        device=device,
        dist="hamming",
        ratio=0.95,
        threshold=160,
    )

    np.savetxt("./output_matches.out", matches.cpu().numpy())

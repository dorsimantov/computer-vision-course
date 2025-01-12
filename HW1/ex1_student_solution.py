"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        a = []
        for (xsrc, ysrc), (xdst, ydst) in zip(match_p_src.T, match_p_dst.T):
            a.append([-xsrc, -ysrc, -1, 0, 0, 0, xsrc * xdst, ysrc * xdst, xdst])
            a.append([0, 0, 0, -xsrc, -ysrc, -1, xsrc * ydst, ysrc * ydst, ydst])
        a = np.array(a)
        new_a = np.dot(a.T, a)
        eigvals, eigvecs = np.linalg.eig(new_a)
        h = eigvecs[:, np.argmin(eigvals)]  # gets smallest eigenvalue
        h = h / h[-1]
        return h.reshape((3, 3))

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        dst_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        for y in range(src_image.shape[:2][0]):
            for x in range(src_image.shape[:2][1]):  # we'll solve it brute force with loops
                src_point = np.array([x, y, 1])
                dst_point = np.dot(homography, src_point)   # projection to the new coordinate system
                dx, dy = dst_point[:2] / dst_point[2]
                dx, dy = int(round(dx)), int(round(dy))
                if 0 <= dx < dst_image_shape[1] and 0 <= dy < dst_image_shape[0]: # we'll check if it's in bounds
                    dst_image[dy, dx] = src_image[y, x]
        return dst_image


    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        dst_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        Hs, Ws = src_image.shape[:2]
        ys, xs = np.meshgrid(np.arange(Hs), np.arange(Ws), indexing='ij')   # creating a meshgrid
        src_points = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs.ravel())])
        proj_points = homography @ src_points    # applying the homography on the meshgrid
        proj_points /= proj_points[2, :]
        xd = np.clip(proj_points[0, :].astype(int), 0, dst_image_shape[1] - 1)   # if not in bounds, removed
        yd = np.clip(proj_points[1, :].astype(int), 0, dst_image_shape[0] - 1)
        dst_image[yd, xd] = src_image[ys.ravel(), xs.ravel()]
        return dst_image


    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""
        # Get number of points N
        num_points = match_p_src.shape[1]
        # Source points -> homogeneous coordinates
        match_p_src_hom = np.vstack((match_p_src, np.ones(num_points)))
        # Transform source points
        match_p_src_hom_transformed = homography @ match_p_src_hom
        # Homogeneous coordinates -> Cartesian (normalize by third row and drop it)
        match_p_src_transformed = match_p_src_hom_transformed[:2] / match_p_src_hom_transformed[2]
        # Find l2 distances between transformed source points and target points
        distances = np.linalg.norm(match_p_src_transformed - match_p_dst, axis=0)
        # Get mask of inliers by filtering out all pairs with distance above given threshold
        inliers = distances <= max_err
        # Number of inliers
        num_inliers = inliers.sum()
        # If no inliers, return dist_mse = 10 ** 9
        if num_inliers == 0:
            dist_mse = 10 ** 9
        # Else, return MSE of distances over inliers only
        else:
            dist_mse = np.mean(np.square(distances[inliers]))
        # Find percentage of inliers
        fit_percent = num_inliers / num_points
        # Return fit_percent and dist_mse
        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""
        # Get number of points N
        num_points = match_p_src.shape[1]
        # Source points -> homogeneous coordinates
        match_p_src_hom = np.vstack((match_p_src, np.ones(num_points)))
        # Transform source points
        match_p_src_hom_transformed = homography @ match_p_src_hom
        # Homogeneous coordinates -> Cartesian (normalize by third row and drop it)
        match_p_src_transformed = match_p_src_hom_transformed[:2] / match_p_src_hom_transformed[2]
        # Find l2 distances between transformed source points and target points
        distances = np.linalg.norm(match_p_src_transformed - match_p_dst, axis=0)
        # Get mask of inliers by filtering out all pairs with distance above given threshold
        inliers = distances <= max_err
        # Choose only the D inliers
        mp_src_meets_model = match_p_src[:, inliers]
        mp_dst_meets_model = match_p_dst[:, inliers]
        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        # w = inliers_percent
        # # t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        # p = 0.99
        # # the minimal probability of points which meets with the model
        # d = 0.5
        # # number of points sufficient to compute the model
        # n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        # k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography
        """INSERT YOUR CODE HERE"""
        # Define the parameters
        w = inliers_percent
        t = max_err
        p = 0.99
        d = 0.5
        n = 4
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1

        # Initialization
        best_model = None
        best_error = np.inf

        # For k iterations
        for _ in range(k):
            # Randomly select n points
            point_indices = np.random.choice(match_p_src.shape[1], n, replace=False)
            src_points = match_p_src[:, point_indices]
            dst_points = match_p_dst[:, point_indices]

            # Compute model using n points
            homography = self.compute_homography_naive(src_points, dst_points)

            # Find fit ratio
            fit_percent, _ = self.test_homography(
                homography,
                src_points,
                dst_points,
                t
            )

            # If fit ratio > d
            if fit_percent > d:
                # Find inliers
                src_inliers, dst_inliers = self.meet_the_model_points(
                    homography,
                    src_points,
                    dst_points,
                    t
                )
                # Re-compute model using all inliers
                homography = self.compute_homography_naive(src_inliers, dst_inliers)
                # If error is best so far, set best homography to current
                _, error = self.test_homography(
                    homography,
                    src_points,
                    dst_points,
                    t
                )
                if error < best_error:
                    best_model = homography
                    best_error = error

        # Finally, return the best homography
        return best_model


    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """
        # (1) Create a mesh-grid of columns and rows of the destination image
        dst_rows, dst_cols = np.meshgrid(np.arange(dst_image_shape[0]), np.arange(dst_image_shape[1]), indexing='ij')

        # (2) Create a set of homogenous coordinates for the destination image using the mesh-grid from (1)
        dst_homogeneous_coords = np.vstack([dst_cols.ravel(), dst_rows.ravel(), np.ones(dst_cols.size)])

        # (3) Compute the corresponding coordinates in the source image using the backward projective homography
        src_homogeneous_coords = backward_projective_homography @ dst_homogeneous_coords
        # Homogeneous -> Cartesian
        src_coords = src_homogeneous_coords[:2] / src_homogeneous_coords[2]

        # (4) Create the mesh-grid of source image coordinates
        src_rows, src_cols = np.meshgrid(np.arange(src_image.shape[0]), np.arange(src_image.shape[1]), indexing='ij')
        src_coords_points = np.vstack((src_cols.ravel(), src_rows.ravel())).T

        # (5) For each color channel (RGB): Use scipy's interpolation.griddata
        # with an appropriate configuration to compute the bi-cubic interpolation of the projected coordinates
        output_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        for channel in range(3):  # For each color channel
            # Apply bicubic interpolation
            src_values = src_image[:, :, channel].ravel()
            interpolated_channel = griddata(
                src_coords_points,
                src_values,
                src_coords.T,
                method='cubic',
                fill_value=0
            )
            output_image[:, :, channel] = interpolated_channel.reshape(dst_image_shape[0], dst_image_shape[1])

        # Finally, return warped image
        return output_image

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """




        # 1) the translation matrix from pads:
        tran_mat = np.array([
        [1, 0, -pad_left],
        [0, 1, -pad_up],
        [0, 0, 1]])
        # 2) compibe the matrices
        hom_and_tran = np.dot(backward_homography, tran_mat)
        # 3) normalize
        final_homography = hom_and_tran / hom_and_tran[2, 2]
        return final_homography

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # (1) Compute the forward homography and the panorama shape.
        homography = self.compute_homography(match_p_src=match_p_src,match_p_dst=match_p_dst,inliers_percent=inliers_percent,max_err=max_err)
        panorama_rows_num, panorama_cols_num, pad_struct = self.find_panorama_shape(src_image=src_image,dst_image=dst_image,homography=homography)
        # (2) Compute the backward homography.
        backward_homography = np.linalg.inv(homography)
        backward_homography = backward_homography/backward_homography[2,2]
        # (3) Add the appropriate translation to the homography so that the
        #     source image will plant in place.
        backward_homography = self.add_translation_to_backward_homography(backward_homography,pad_struct.pad_left,pad_struct.pad_up)
        # (4) Compute the backward warping with the appropriate translation.
        import matplotlib.pyplot as plt

        backward_warping = self.compute_backward_mapping(backward_homography, src_image, (panorama_rows_num, panorama_cols_num,3))
        plt.imshow(backward_warping)
        plt.show()
        # (5) Create an empty panorama image and plant there the
        #     destination image.
        img_panorama = np.zeros((panorama_rows_num, panorama_cols_num, 3), dtype=src_image.dtype)
        small_h, small_w = dst_image.shape[:2]
        large_h, large_w = img_panorama.shape[:2]
        y_start = pad_struct.pad_up  # Subtract y_offset to move up
        y_end = y_start + small_h
        x_start = pad_struct.pad_left  # Add x_offset to move right
        x_end = x_start + small_w


        # (6) place the backward warped image in the indices where the panorama
        #     image is zero.
        img_panorama[0:backward_warping.shape[0],
        0:backward_warping.shape[1]] = backward_warping

        img_panorama[y_start:y_end, x_start:x_end] = dst_image
        """img_panorama[pad_struct.pad_up:pad_struct.pad_up + dst_image.shape[0],
        pad_struct.pad_left:pad_struct.pad_left + dst_image.shape[1]] = dst_image"""
        plt.imshow(img_panorama)
        plt.show()

        return np.clip(img_panorama, 0, 255).astype(np.uint8)
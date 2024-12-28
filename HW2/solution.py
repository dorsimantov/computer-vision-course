"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        half_win = win_size // 2
        padded_left = np.pad(left_image,
                             ((half_win, half_win),
                              (half_win, half_win),
                              (0, 0)),
                             mode='constant')
        padded_right = np.pad(right_image,
                              ((half_win, half_win),
                               (half_win, half_win),
                               (0, 0)),
                              mode='constant')

        # Iterate through each pixel
        for row in range(num_of_rows):
            for col in range(num_of_cols):
                # Extract window from left image
                left_window = padded_left[row:row + win_size,
                              col:col + win_size, :]

                # Compute SSD for each disparity value
                for d_idx, d in enumerate(disparity_values):
                    # Shifted right image window
                    if(num_of_cols + half_win*2 < col + d + win_size or win_size > col + d + win_size):
                        ssdd_tensor[row, col, d_idx] = 100000
                        continue
                    right_window = padded_right[row:row + win_size,
                                   col + d:col + d + win_size, :]

                    # Compute sum of squared differences
                    ssd = np.sum((left_window - right_window) ** 2)

                    # Store in tensor
                    ssdd_tensor[row, col, d_idx] = ssd
        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # Create output matrix of the same height and width as the input tensor
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))

        # For each pixel, find the index of the minimum SSD value
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)

        # Adjust to get the correct disparity value
        # (since argmin returns 0-based index, we need to map it to the actual disparity range)
        label_no_smooth -= (ssdd_tensor.shape[2] - 1) // 2

        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))

        # Get the shape of the input slice
        dsp_range, width = c_slice.shape

        # Initialize the output matrix
        l_slice = np.zeros((2 * dsp_range + 1, width))

        # Iterate through each column
        for col in range(width):
            # Get the current column
            curr_col = c_slice[:, col]

            # Normalize the current column with the minimum value from the previous column
            if col > 0:
                prev_col = c_slice[:, col - 1]
                curr_col = curr_col - prev_col.min()

            # Calculate the score for each disparity value in the current column
            for d in range(dsp_range):
                if col == 0:
                    l_slice[d, col] = curr_col[d]
                    continue
                # Option 1: Score from the previous column for the same d value
                option1 = l_slice[d, col - 1]

                # Option 2: Score from the previous column with disparity value deviating by ±1, plus penalty p1
                """option2 = l_slice[max(0, d - 1), col - 1] + p1 if col > 0 else 0
                option2 = min(option2, l_slice[min(2 * dsp_range, d + 1), col - 1] + p1)"""
                option2 = p1 + min(l_slice[d-1,col-1] if col > 0 else 0, l_slice[d+1,col-1] if col > 0 else 0)

                # Option 3: Score from the previous column with disparity value deviating by more than 2, plus penalty p2
                """option3 = l_slice[max(0, d - 2), col - 1] + p2 if col > 0 else 0
                option3 = min(option3, l_slice[min(2 * dsp_range, d + 2), col - 1] + p2)"""

                option3 = p2 + min(l_slice[:d-1,col-1] if d > 1 else [] + l_slice[d+1:,col-1] if d < dsp_range else [])

                # Choose the minimum score and add the current column cost
                l_slice[d, col] = curr_col[d] + min(option1, option2, option3)

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)
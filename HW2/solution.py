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
        """Compute the SSDD distances tensor."""
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))

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

        for row in range(num_of_rows):
            for col in range(num_of_cols):
                left_window = padded_left[row:row + win_size,
                              col:col + win_size, :]

                for d_idx, d in enumerate(disparity_values):
                    if (col + d + win_size > num_of_cols + 2 * half_win or
                            col + d < 0):
                        ssdd_tensor[row, col, d_idx] = float('inf')
                        continue

                    right_window = padded_right[row:row + win_size,
                                   col + d:col + d + win_size, :]
                    ssd = np.sum((left_window - right_window) ** 2)
                    ssdd_tensor[row, col, d_idx] = ssd

        # Normalize the tensor
        valid_values = ssdd_tensor[ssdd_tensor != float('inf')]
        if len(valid_values) > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            ssdd_tensor[ssdd_tensor != float('inf')] = (
                    (ssdd_tensor[ssdd_tensor != float('inf')] - min_val) /
                    (max_val - min_val) * 255.0
            )
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
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
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
        num_labels, width = c_slice.shape
        l_slice = np.zeros((num_labels, width))
        l_slice[:, 0] = c_slice[:, 0]
        for w in range(1, width):
            prev_costs = l_slice[:, w - 1]
            # Calculate all possible transitions
            min_prev = np.min(prev_costs)
            d1_costs = np.roll(prev_costs, 1)
            d1_costs[0] = float('inf')
            d2_costs = np.roll(prev_costs, -1)
            d2_costs[-1] = float('inf')
            min_cost = np.minimum.reduce([
                prev_costs,  # No change in disparity
                d1_costs + p1,  # Change by +1
                d2_costs + p1,  # Change by -1
                np.full_like(prev_costs, min_prev + p2)  # Larger changes
            ])
            l_slice[:, w] = min_cost + c_slice[:, w]
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
        height, width, num_disp = ssdd_tensor.shape
        l = np.zeros_like(ssdd_tensor)

        # Process each row
        for y in range(height):
            l[y, :, :] = self.dp_grade_slice(
                ssdd_tensor[y, :, :].T, p1, p2
            ).T

        return self.naive_labeling(l)

    def sides_up_down(self,
                      ssdd_tensor: np.ndarray,
                      p1: float,
                      p2: float) -> np.ndarray:
        rows = ssdd_tensor.shape[0]
        result = np.zeros_like(ssdd_tensor)
        for row_idx in range(rows):
            horizontal_slice = ssdd_tensor[row_idx, :, :].T
            result[row_idx, :, :] = self.dp_grade_slice(horizontal_slice, p1, p2).T
        return result

    def sides_diagonals(self,
                        ssdd_tensor: np.ndarray,
                        p1: float,
                        p2: float) -> np.ndarray:
        rows, cols, labels = ssdd_tensor.shape
        dim_diff = cols - rows
        result = np.zeros_like(ssdd_tensor)
        result[-1, 0] = ssdd_tensor[-1, 0]
        result[0, -1] = ssdd_tensor[0, -1]
        start_diag = 2 - rows
        end_diag = cols - 1
        for diagonal_offset in range(start_diag, end_diag):
            current_diagonal = np.diagonal(ssdd_tensor, offset=diagonal_offset)
            smoothed_diagonal = self.dp_grade_slice(current_diagonal, p1, p2).T
            for label_idx in range(labels):
                if diagonal_offset <= 0:        # Region before top-left corner
                    result[:, :rows, label_idx] += np.diag(
                        smoothed_diagonal[:, label_idx],
                        k=diagonal_offset)
                elif diagonal_offset + rows < cols:     # Region between corners
                    result[:, diagonal_offset:rows + diagonal_offset, label_idx] += np.diag(
                        smoothed_diagonal[:, label_idx])
                else:       # Region after bottom-right corner
                    result[:, -rows:, label_idx] += np.diag(
                        smoothed_diagonal[:, label_idx],
                        k=diagonal_offset - dim_diff)
        return result

    def compute_single_direction(self, ssdd_tensor, p1, p2, direction):
        def transform_tensor(tensor, direction):
            if direction == 1:
                return tensor.copy(), lambda x: x
            elif direction == 5:
                return np.flip(tensor, axis=1), lambda x: np.flip(x, axis=1)
            elif direction == 3:
                return np.transpose(tensor, axes=[1, 0, 2]), lambda x: np.transpose(x, axes=[1, 0, 2])
            elif direction == 7:
                return np.flip(np.transpose(tensor, axes=[1, 0, 2]), axis=1), lambda x: np.transpose(np.flip(x, axis=1),
                                                                                                     axes=[1, 0, 2])
            elif direction == 2:
                return tensor.copy(), lambda x: x
            elif direction == 4:
                return np.flip(tensor, axis=1), lambda x: np.flip(x, axis=1)
            elif direction == 6:
                return np.flip(tensor, axis=(0, 1)), lambda x: np.flip(x, axis=(0, 1))
            elif direction == 8:
                return np.flip(tensor, axis=0), lambda x: np.flip(x, axis=0)
            else:
                return None, None  # Invalid direction

        modified_tensor, inverse_transform = transform_tensor(ssdd_tensor, direction)
        if direction in {1, 3, 5, 7}:  # Orthogonal directions
            processed_tensor = self.sides_up_down(modified_tensor, p1, p2)
        else:  # Diagonal directions
            processed_tensor = self.sides_diagonals(modified_tensor, 0.5, 3)
        return inverse_transform(processed_tensor)

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
        NUM_DIRECTIONS = 8
        direction_to_depth = {}
        for direction in range(1, NUM_DIRECTIONS + 1):
            smoothed_costs = self.compute_single_direction(
                ssdd_tensor=ssdd_tensor,
                p1=p1,
                p2=p2,
                direction=direction)
            direction_to_depth[direction] = self.naive_labeling(smoothed_costs)
        return direction_to_depth

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
        NUM_DIRECTIONS = 8
        aggregated_cost = np.zeros_like(ssdd_tensor)
        for direction in range(1, NUM_DIRECTIONS + 1):      # Process each direction and accumulate results
            direction_cost = self.compute_single_direction(
                ssdd_tensor=ssdd_tensor,
                p1=p1,
                p2=p2,
                direction=direction)
            aggregated_cost += direction_cost
        averaged_cost = aggregated_cost / NUM_DIRECTIONS
        return self.naive_labeling(averaged_cost)
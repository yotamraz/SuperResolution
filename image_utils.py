
import torch
import numpy as np


def image_to_tensor(image: np.ndarray, range_norm: bool = True, half_prec: bool = True) -> torch.Tensor:
    """
    Converts a NumPy array representing an image to a PyTorch Tensor.

    Parameters:
    - image (np.ndarray): A NumPy array representing the image. The shape should be (height, width, channels).
    - range_norm (bool, optional): If True, the image data will be scaled from [0, 1] to [-1, 1]. Default is True.
    - half_prec (bool, optional): If True, the image data type will be converted to torch.half. Default is True.

    Returns:
    - torch.Tensor: A PyTorch Tensor representing the image. The shape is (channels, height, width).
    """

    # Convert image data type to Tensor data type, channels first
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half_prec:
        tensor = tensor.half()

    return tensor



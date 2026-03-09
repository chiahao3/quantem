"""
Utility functions used for benchmarking
"""

import torch
import os
import numpy as np

def time_sync(device=None):
    # PyTorch doesn't have a direct exposed API to check the selected default device 
    # so we'll be checking these .is_available() just to prevent error.
    # Luckily these checks won't really affect the performance.
    
    from time import perf_counter
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    # Check if MPS (Metal Performance Shaders) is available (macOS only)
    elif torch.backends.mps.is_available():
        torch.mps.synchronize(device)
    
    # Measure the time
    t = perf_counter()
    return t

def load_raw(file_path, shape, dtype=np.float32, offset=0, gap=1024):
    """Loads a raw binary file containing interleaved image data and gaps.

    This implementation uses a custom `numpy.dtype` with `np.fromfile` for 
    fast I/O performance, extracting only the valid data regions and skipping 
    the specified byte gaps between frames. Note that custom processed raw 
    data might have a gap of 0.

    Args:
        file_path (str): The path to the raw binary file.
        shape (tuple of int): The expected shape of the data in the format 
            (N, height, width), where N is the number of frames.
        dtype (data-type, optional): The NumPy data type of the image pixels. 
            Defaults to np.float32.
        offset (int, optional): The number of bytes to skip at the beginning 
            of the file. Defaults to 0.
        gap (int, optional): The number of gap bytes to skip between each 
            image frame. Defaults to 1024.

    Returns:
        numpy.ndarray: An array of the extracted data with the specified shape 
        and dtype.

    Raises:
        ValueError: If the actual file size does not match the expected size 
            calculated from the inputs.
    """
    # shape = (N, height, width)
    # np.fromfile with custom dtype is faster than the np.read and np.frombuffer
    # This implementaiton is also roughly 2x faster (10sec vs 20sec) than load_hdf5 with a 128x128x128x128 (1GB) EMPAD dataset
    # Note that for custom processed empad2 raw there might be no gap between the images
    N, height, width = shape

    # Verify file size first
    expected_size = offset + N * (height * width * dtype().itemsize + gap)
    actual_size = os.path.getsize(file_path)

    if actual_size != expected_size:
        raise ValueError(f"Mismatch in expected ({expected_size} bytes = offset + N * (height * width * 4 + gap)) vs. actual ({actual_size} bytes) file size! Check your loading configurations!")
    
    # Define the custom dtype to include both data and gap
    custom_dtype = np.dtype([
        ('data', dtype, (height, width)),
        ('gap', np.uint8, gap)  # uint8 means 1 byte per gap element
    ])

    # Read the entire file using the custom dtype
    with open(file_path, 'rb') as f:
        f.seek(offset)
        raw_data = np.fromfile(f, dtype=custom_dtype, count=N)

    # Extract just the 'data' part (ignoring the gaps)
    data = raw_data['data']
    return data
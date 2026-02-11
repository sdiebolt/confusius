"""Convert Pyfus RAW IQ data to ConfUSIus Zarr format."""

from pathlib import Path
from typing import Any
import numpy as np
import zarr
from pyfus.io import load_iq
from rich.progress import track


def convert_raw_to_zarr(
    raw_path: str | Path,
    output_path: str | Path,
    volumes_per_chunk: int | None = None,
    batch_size: int = 100,
    overwrite: bool = False,
    show_progress: bool = True,
) -> zarr.Group:
    """Convert a RAW IQ file to Zarr format compatible with ConfUSIus.

    Parameters
    ----------
    raw_path : str or Path
        Path to the RAW IQ file.
    output_path : str or Path
        Path where the Zarr group will be saved.
    volumes_per_chunk : int, optional
        Number of volumes to include in each Zarr chunk. If not specified,
        defaults to the number of volumes per block.
    batch_size : int, default: 100
        Number of blocks to process in each batch to avoid memory issues.
    overwrite : bool, default: False
        Whether to overwrite existing Zarr group.
    show_progress : bool, default: True
        Whether to show progress during conversion.

    Returns
    -------
    zarr.Group
        The created Zarr group containing the IQ data and coordinates.
    """
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    iq = load_iq(raw_path)

    n_blocks = iq.nblock
    n_volumes_per_block = iq.nvolume
    n_total_volumes = n_blocks * n_volumes_per_block

    if volumes_per_chunk is None:
        volumes_per_chunk = n_volumes_per_block

    block_shape = iq.shape[:-1]
    n_x = block_shape[0]
    n_y = block_shape[1]
    n_z = block_shape[2]

    output_shape = (n_total_volumes, n_z, n_y, n_x)
    chunks = (volumes_per_chunk, n_z, n_y, n_x)

    zarr_group = zarr.open_group(output_path, mode="w" if overwrite else "w-")

    dim_names = ["time", "z", "y", "x"]

    zarr_iq = zarr_group.create_array(
        "iq",
        shape=output_shape,
        chunks=chunks,
        dtype=iq._data.dtype,
        dimension_names=dim_names,
    )

    time_values = iq.time.flatten()
    zarr_group.create_array("time", data=time_values, dimension_names=["time"])
    zarr_group["time"].attrs["units"] = "s"
    zarr_group["time"].attrs["long_name"] = "Time"

    z_axis = np.arange(n_z, dtype=np.float64) * iq.voxdim[2] * 1e3
    zarr_group.create_array("z", data=z_axis, dimension_names=["z"])
    zarr_group["z"].attrs["units"] = "mm"
    zarr_group["z"].attrs["long_name"] = "Depth"

    y_axis = np.arange(n_y, dtype=np.float64) * iq.voxdim[1] * 1e3
    zarr_group.create_array("y", data=y_axis, dimension_names=["y"])
    zarr_group["y"].attrs["units"] = "mm"
    zarr_group["y"].attrs["long_name"] = "Elevation"

    x_axis = np.arange(n_x, dtype=np.float64) * iq.voxdim[0] * 1e3
    zarr_group.create_array("x", data=x_axis, dimension_names=["x"])
    zarr_group["x"].attrs["units"] = "mm"
    zarr_group["x"].attrs["long_name"] = "Lateral"

    zarr_group.attrs["voxdim"] = [
        float(iq.voxdim[0] * 1e3),
        float(iq.voxdim[1] * 1e3),
        float(iq.voxdim[2] * 1e3),
    ]
    zarr_group.attrs["transmit_frequency"] = iq.ultrasound_frequency
    zarr_group.attrs["sound_velocity"] = iq.sound_velocity
    zarr_group.attrs["compound_sampling_frequency"] = iq.volume_framerate
    zarr_group.attrs["modality"] = iq.modality
    zarr_group.attrs["aperture"] = iq.aperture.tolist()
    zarr_group.attrs["depth"] = iq.depth.tolist()

    n_plane_wave_angles = len(iq.plane_wave_angles)
    if n_plane_wave_angles > 1:
        zarr_group.attrs["plane_wave_angles"] = iq.plane_wave_angles.tolist()

    n_batches = (n_blocks + batch_size - 1) // batch_size
    batches = range(0, n_blocks, batch_size)

    if show_progress:
        batches = track(batches, description="Converting IQ data to Zarr...")

    for start_block in batches:
        end_block = min(start_block + batch_size, n_blocks)
        n_blocks_in_batch = end_block - start_block

        batch_data = iq._data[:, :, :, :, :, start_block:end_block]

        batch_data = np.squeeze(batch_data, axis=(1, 3))
        batch_data = np.moveaxis(batch_data, -1, 2)
        batch_data = batch_data.reshape(n_x, n_z, n_y, -1)
        batch_data = np.transpose(batch_data, (3, 1, 2, 0))

        start_vol = start_block * n_volumes_per_block
        end_vol = start_vol + batch_data.shape[0]

        zarr_iq[start_vol:end_vol] = batch_data

    return zarr_group


if __name__ == "__main__":
    import shutil

    raw_path = (
        "/home/sdiebolt/Documents/Work/physmed/awake-fusi-denoising/data/sub-tatooine/"
        "ses-20221205/fus/sub-tatooine_ses-20221205_task-anesthetized_pd2dt.raw"
    )
    output_path = "/home/sdiebolt/Documents/Work/physmed/awake-fusi-denoising/data/sub-tatooine/ses-20221205/fus/iq.zarr"

    if Path(output_path).exists():
        shutil.rmtree(output_path)

    convert_raw_to_zarr(raw_path, output_path, batch_size=100, show_progress=True)
    print(f"Converted {raw_path} to {output_path}")

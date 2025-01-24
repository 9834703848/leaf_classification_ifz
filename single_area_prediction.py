import richdem as rd
import numpy as np
import rasterio
import rasterio.warp
import os
import statistics




def write_array_in_chunks(output_path, data_array, meta, chunk_size=1000):
    """
    Writes a large array to a file in chunks to avoid memory errors.

    Parameters:
    - output_path: The path where the raster file will be saved.
    - data_array: The 2D array of data to write.
    - meta: Metadata for the raster file.
    - chunk_size: The size of each chunk (number of rows) to write at a time.
    """
    # Update metadata for float32 and single band
    meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})

    with rasterio.open(output_path, "w", **meta) as dest:
        rows, cols = data_array.shape

        # Write in chunks
        for row_start in range(0, rows, chunk_size):
            row_end = min(row_start + chunk_size, rows)
            dest.write(data_array[row_start:row_end, :], 1, window=((row_start, row_end), (0, cols)))




def single_area_prediction(DEM_PATH, DEM_file, area_file, temp_utm_file='utm_file.tif', temp_area_file='tmp_file.tif', epsg_utm="EPSG:32632", epsg_wgs84="EPSG:4326"):
    """
    Processes a DEM file to calculate the area gradient and project it from UTM to WGS84.

    Parameters:
    - DEM_PATH: Path to the directory containing the DEM file.
    - DEM_file: The DEM file name.
    - area_file: The output area file name.
    - temp_utm_file: Temporary file for UTM transformation.
    - temp_area_file: Temporary file for area calculation.
    - epsg_utm: EPSG code for UTM projection.
    - epsg_wgs84: EPSG code for WGS84 projection.

    Returns:
    - None
    """
    # Construct paths
    input_raster_path = os.path.join(DEM_PATH, DEM_file)
    path_utm = os.path.join(DEM_PATH, temp_utm_file)
    area_output_path = os.path.join(DEM_PATH, temp_area_file)
    final_area_path = os.path.join(DEM_PATH, area_file)

    # Read and reproject the DEM file to UTM
    with rasterio.open(input_raster_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, epsg_utm, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': epsg_utm,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(path_utm, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=epsg_utm,
                    resampling=rasterio.enums.Resampling.nearest
                )

    # Load UTM transformed DEM and calculate the area gradient
    with rasterio.open(path_utm) as dem:
        resx_utm = dem.res[0]
        resy_utm = dem.res[1]
        dem_array = dem.read(1)

    area_array = cgrad_chunked(dem_array=dem_array, resx_utm=resx_utm, resy_utm=resy_utm, area=True)

    # Calculate cell size and area
    coefficient = 1
    cell_size = statistics.mean([resx_utm, resy_utm]) * coefficient
    cell_area = resx_utm * resy_utm * coefficient * coefficient
    g = area_array / cell_area

    # Save the area array to a temporary file
    out_meta = dem.meta.copy()
    out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
    write_array_in_chunks(area_output_path, g, out_meta, chunk_size=1000)

    # Reproject the area file from UTM to WGS84
    with rasterio.open(area_output_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, epsg_wgs84, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': epsg_wgs84,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(final_area_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=epsg_wgs84,
                    resampling=rasterio.enums.Resampling.nearest
                )

    # Clean up temporary files
    os.remove(area_output_path)

import numpy as np

def cgrad_chunked(dem_array, resx_utm=0, resy_utm=0, area=False, chunk_size=1000):
    """
    Calculates the gradient of a DEM array using chunk processing to avoid memory errors.

    Parameters:
    - dem_array: 2D numpy array of DEM values.
    - resx_utm: Resolution in the x direction (UTM).
    - resy_utm: Resolution in the y direction (UTM).
    - area: Boolean to indicate if area calculation is required.
    - chunk_size: Size of chunks to process at a time.

    Returns:
    - Gradient or normalized gradient array.
    """
    dly = resx_utm
    dlx = resy_utm
    mm = dem_array
    rows, cols = mm.shape

    # Create an empty array to store the results
    cellArea = np.zeros((rows, cols), dtype=np.float32)

    # Process in chunks to avoid memory errors
    for row_start in range(0, rows, chunk_size):
        for col_start in range(0, cols, chunk_size):
            # Define chunk boundaries
            row_end = min(row_start + chunk_size, rows)
            col_end = min(col_start + chunk_size, cols)

            # Extract the chunk
            mm_chunk = mm[row_start:row_end, col_start:col_end]

            # Compute gradient within the chunk
            cellgr = np.zeros((mm_chunk.shape[0], mm_chunk.shape[1], 3), dtype=np.float32)

            # Handle edge cases within chunks
            if row_start > 0 and col_start > 0:
                md = mm[row_start - 1:row_end - 1, col_start:col_end]
                mr = mm[row_start:row_end, col_start - 1:col_end - 1]
                mrd = mm[row_start:row_end, col_start:col_end]
                cellgr[:, :, 1] = 0.5 * dlx * (mm_chunk + md - mr - mrd)
                cellgr[:, :, 0] = 0.5 * dly * (mm_chunk - md + mr - mrd)
                cellgr[:, :, 2] = dlx * dly

            # Calculate the area for the chunk
            chunk_area = np.sqrt(
                np.power(cellgr[:, :, 0], 2) + np.power(cellgr[:, :, 1], 2) + np.power(cellgr[:, :, 2], 2)
            )
            cellArea[row_start:row_end, col_start:col_end] = chunk_area

    # Normalize if area is not required
    output = cellArea if area else cellArea / np.max(cellArea)

    return output


# Example usage:
# single_area_prediction(DEM_PATH='path_to_dem', DEM_file='20230911.tif', area_file='AREA_20230911.tif')

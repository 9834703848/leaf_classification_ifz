import richdem as rd
import numpy as np
import rasterio
import rasterio.warp
import os

def single_slope_calculation(DEM_PATH, DEM_file, slope_file, epsg_utm="EPSG:32632", epsg_wgs84="EPSG:4326", chunk_size=1000):
    """
    Processes a DEM file to calculate the slope in degrees and projects it from UTM to WGS84 in memory-efficient chunks.

    Parameters:
    - DEM_PATH: Path to the directory containing the DEM file.
    - DEM_file: The DEM file name.
    - slope_file: The output slope file name.
    - epsg_utm: EPSG code for UTM projection.
    - epsg_wgs84: EPSG code for WGS84 projection.
    - chunk_size: Number of rows to process at a time for memory efficiency.

    Returns:
    - None
    """
    # Construct paths
    input_raster_path = os.path.join(DEM_PATH, DEM_file)
    path_utm = os.path.join(DEM_PATH, slope_file)
    temp_slope_path = os.path.join(DEM_PATH, "temp_slope.tif")

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

    # Load UTM transformed DEM and calculate slope in chunks
    with rasterio.open(path_utm) as src:
        meta = src.meta.copy()
        meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})

        # Create an empty slope raster
        with rasterio.open(temp_slope_path, 'w', **meta) as dest:
            for row_start in range(0, src.height, chunk_size):
                row_end = min(row_start + chunk_size, src.height)
                window = rasterio.windows.Window(0, row_start, src.width, row_end - row_start)

                # Read the chunk
                dem_chunk = src.read(1, window=window)

                # Calculate slope for the chunk
                dem_chunk = rd.rdarray(dem_chunk, no_data=np.nan)
                slope_chunk = rd.TerrainAttribute(dem_chunk, attrib='slope_degrees')
                slope_chunk[slope_chunk < 0] = np.nan

                # Write the slope chunk back to the temporary file
                dest.write(slope_chunk, 1, window=window)

    # Reproject the slope file from UTM to WGS84 in chunks
    with rasterio.open(temp_slope_path) as src:
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

        with rasterio.open(os.path.join(DEM_PATH, slope_file), 'w', **kwargs) as dst:
            for row_start in range(0, src.height, chunk_size):
                row_end = min(row_start + chunk_size, src.height)
                window = rasterio.windows.Window(0, row_start, src.width, row_end - row_start)

                # Read the slope chunk
                slope_chunk = src.read(1, window=window)

                # Reproject and write the slope chunk to the final output
                rasterio.warp.reproject(
                    source=slope_chunk,
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=epsg_wgs84,
                    resampling=rasterio.enums.Resampling.nearest,
                    src_window=window,
                    dst_window=window
                )

    # Clean up temporary files
    os.remove(temp_slope_path)

# Example usage:
# single_slope_calculation(DEM_PATH='path_to_dem', DEM_file='20230911.tif', slope_file='SLOPE_20230911.tif')

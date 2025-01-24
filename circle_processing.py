from leaf_utils import *
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from sklearn.impute import SimpleImputer
import random

def circle_processing(params):
    # Unpack parameters from the dictionary
    model = params['model']
    ortho_file_path = params['ortho_file_path']
    area_path = params['area_path']
    slope_path = params['slope_path']
    pipeline = params['pipeline']
    bands = params['bands']
    shape_temp = params['shape_temp']
    UTMzone = params['UTMzone']
    profile = params['profile']
    n_ins = params['n_ins']
    F1 = params['F1']
    RESULTS_FOLDER = params['RESULTS_FOLDER']

    orthomosaic_crs = profile['crs']
    utm_crs = f'+proj=utm +zone={UTMzone} +ellps=WGS84'

    # Process leaf geometries
    leaf_gdf = shape_temp.to_crs(utm_crs)

    # Generate non-overlapping circles
    num_circles_per_polygon = n_ins * 3
    circle_radius = 0.05
    circle_list = []

    for polygon in leaf_gdf.geometry:
        circles = generate_non_overlapping_circles_within_polygon(polygon, num_circles_per_polygon, circle_radius)
        circle_list.extend(circles)

    # Create a GeoDataFrame for the circles
    circle_gdf = gpd.GeoDataFrame(geometry=circle_list, crs=utm_crs).to_crs(orthomosaic_crs)
    circle_gdf['circle_id'] = range(1, len(circle_gdf) + 1)

    # Crop and mask the orthomosaic based on circles
    C1, C1_transform = crop_raster(ortho_file_path, circle_gdf.geometry)
    T1, T1_transform = C1, C1_transform
    nr, nc = T1.shape[1], T1.shape[2]

    # Initialize table_bands
    table_bands = None

    # Process each band
    for b in range(len(bands)):
        IDX = T1[b]
        if np.any(np.isinf(np.abs(IDX))):
            IDX[np.isinf(IDX)] = np.nan
            IDX = np.log10(IDX)
            IDX[np.isinf(IDX)] = np.nan
            IDX = np.reshape(IDX, (nr, nc))
        
        IDX = IDX.flatten()
        table_bands = pd.concat([table_bands, pd.Series(IDX, name=bands[b])], axis=1) if table_bands is not None else pd.DataFrame(IDX, columns=[bands[b]])

    # Handle complete cases
    table_bands = table_bands.dropna()
    pos = table_bands.index
    NA_vec = np.ones(nr * nc, dtype=bool)
    NA_vec[pos] = False

    # Create raster data
    raster_data = [np.zeros((1, len(pos)))]
    for j in range(len(bands)):
        band_data = table_bands.iloc[:, j].values
        band_raster = np.zeros((1, len(pos)))
        band_raster[0, :] = band_data
        raster_data.append(band_raster)

    # Stack raster data
    raster_data = np.stack(raster_data, axis=0)

    # Process indices
    index_var = pipeline["indices"]
    table_index = []
    functions_r = {'MSAVIhyper': MSAVIhy, 'GVIMSS': GVIMSS, 'D678500': D678500, 'NSVDI': NSVDI}

    for idx in index_var:
        IDX = functions_r[idx](np.array(bands), raster_data)
        IDX = np.nan_to_num(IDX, nan=np.log10(IDX))
        table_index.append(IDX.ravel())

    table_index = np.array(table_index).T

    if "angle2cam" in pipeline["coef"] or "angle2sun" in pipeline["coef"]:
        angle2cam = process_angle_raster(slope_path, circle_gdf['geometry'], T1, T1_transform, profile, pos)
        if np.isnan(angle2cam).all():
                    # Replace all NaN values with 0
                    angle2cam = np.zeros_like(angle2cam)

    # Prepare data
    data = np.hstack((table_bands.iloc[:, :5], table_index, angle2cam.reshape(-1, 1)))
    data = pd.DataFrame(data, columns=pipeline["coef"])
    ss = np.ones_like(data)
    ss[np.isnan(data)] = 0

    pos_non_na = np.where(ss != 1)
    NA_vec_new = np.full(len(data), False)
    NA_vec_new[np.unique(pos_non_na[0])] = True
    NA_vec[pos[NA_vec_new]] = True

    # Impute and standardize data
    data = np.array(data.loc[:, pipeline["coef"]])
    data = data.flatten().astype(float).reshape(-1, len(pipeline['m']))
    data_imputed = SimpleImputer(strategy='mean').fit_transform(data)
    data_standardized = (data_imputed - np.array(pipeline['m'])) / np.array(pipeline['s'])

    # Predict with the model
    pred = model.predict(data_standardized)
    test_pred_grid = np.argmax(pred, axis=-1)
    mask_pred_circle = np.full((nr * nc), np.nan)
    mask_pred_circle[pos] = test_pred_grid

    # Reshape mask
    mask_pred_circle = mask_pred_circle.reshape((nr, nc))

    # Save the mask as a GeoTIFF
    with rasterio.Env():
        mask_profile = {
            'width': mask_pred_circle.shape[1],
            'height': mask_pred_circle.shape[0],
            'count': 1,
            'transform': T1_transform,
            'crs': profile['crs'],
            'nodata': np.nan,
            'dtype': 'float32'
        }
        with rasterio.open(f"{RESULTS_FOLDER}/circle_mask.tif", 'w', **mask_profile) as dst:
            dst.write(mask_pred_circle, 1)

    rr = rasterize_shapes(circle_gdf, mask_pred_circle.shape, T1_transform)
    rr = np.reshape(rr, (nr, nc))
    rr[np.isnan(rr)] = 0
    rr_ed = rr - erode(rr, F1)

    # Process circles
    seqleaf = np.arange(np.nanmin(rr), np.nanmax(rr))
    rleaf_new = np.random.randint(1, 5, size=(rr.shape[0], rr.shape[1])).astype(str)
    hex_list = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(circle_gdf))]
    rleaf_new[rleaf_new == np.nan] = '#000000'
    for idx, val in enumerate(seqleaf):
        rleaf_new[rr == val] = hex_list[idx]

    instance_vector = rleaf_new.flatten()
    cluster_vector = clustering_mask(mask_pred_circle)
    P1 = process_angle_raster_2(area_path, circle_gdf['geometry'], T1, T1_transform, profile)
    size_vector = P1.flatten()

    # Return results as a dictionary
    return {
        'mask_pred_circle': mask_pred_circle,
        'cluster_vector': cluster_vector,
        'instance_vector': instance_vector,
        'size_vector': size_vector,
        'circle_gdf': circle_gdf,
        'rr_ed': rr_ed,
        'T1': T1
    }



# Further use other results as needed

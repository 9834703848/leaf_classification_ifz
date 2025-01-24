import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio, random
from sklearn.impute import SimpleImputer
from leaf_utils import *

def process_condition_plot(params):
    # Unpack parameters from the dictionary
 
    S1 = params['S1']
    plot_id_temp = params['plot_id_temp']
    RESULTS_FOLDER_ROOT = params['RESULTS_FOLDER_ROOT']
    INSTANCE_FORM = params['INSTANCE_FORM']
    shape_leaf = params['shape_leaf']
    shape_temp = params['shape_temp']
    ortho_file_path = params['ortho_file_path']
    area_path = params['area_path']
    rmux_data_t = params['rmux_data_t']
    rmuy_data_t = params['rmuy_data_t']
    n_ins_test = params['n_ins_test']
    MAX_AREA_LEAF = params['MAX_AREA_LEAF']
    bands = params['bands']
    pipeline = params['pipeline']
    model = params['model']
    class_output = params['class_output']
    min_pix_pred = params['min_pix_pred']
    th_erosion = params['th_erosion']
    F1 = params['F1']
    slope_path = params['slope_path']
    profile = params['profile']
    n_ins = params['n_ins']

    # Check the condition
    condition_plot = not (np.count_nonzero(S1[4] == 0) / S1[0].size * 100 > 5)

    if condition_plot:
        nc, nr = S1.shape[2], S1.shape[1]

        # Creating a plot folder
        folder_name_temp = f"{plot_id_temp:04d}"
        RESULTS_FOLDER = os.path.join(RESULTS_FOLDER_ROOT, folder_name_temp)
        os.makedirs(RESULTS_FOLDER, exist_ok=True)

        # LEAF INSTANCE processing
        if INSTANCE_FORM in ["LEAF", "BOTH"]:
            options_warn = np.seterr(all='ignore')
            plot_leaf = gpd.sjoin(shape_leaf, shape_temp, how='inner', predicate='intersects')
            id_leaf_list = plot_leaf.index.tolist()
            plot_leaf = shape_leaf.loc[id_leaf_list]

            if len(plot_leaf) != 0:
                T1, _ = crop_raster(ortho_file_path, plot_leaf['geometry'])
                nc, nr = T1.shape[1], T1.shape[2]
                pix_size, _ = crop_raster(area_path, plot_leaf['geometry'])
                mask_area = np.nan_to_num(pix_size)
                mask_area[mask_area != 0] = 1
                pix_size[mask_area == 0] = np.nan
                area_leaf_vector = []
                pixel_leaf_vector = []
                error_vector = []

                for leaf in range(len(id_leaf_list)):
                    single_leaf = shape_leaf.loc[[id_leaf_list[leaf]]]
                    try:
                        L1, _ = crop_raster(area_path, single_leaf['geometry'])
                        area_t = np.nansum(L1) * rmux_data_t * rmuy_data_t * 100 * 100
                        area_leaf_vector.append(area_t)
                    except Exception as e:
                        error_vector.append(leaf)

                id_leaf_list = [id_leaf_list[i] for i in range(len(id_leaf_list)) if i not in error_vector]
                qnt = np.nanquantile(area_leaf_vector, [0.1, 0.9])
                first_sample_size = min(n_ins_test, len([x for x in area_leaf_vector if qnt[0] < x < qnt[1]]))
                first_sample_size = int(first_sample_size)
                id_leaf_sel = np.random.choice(
                    [id_leaf_list[i] for i in range(len(area_leaf_vector)) if qnt[0] < area_leaf_vector[i] < qnt[1]],
                    first_sample_size, replace=False
                )

                new_plot_leaf = shape_leaf.iloc[id_leaf_sel]
                LEAF_CONDITION = len(new_plot_leaf) != 0

                if LEAF_CONDITION:
                    print('Processing LEAF_CONDITION')
                    T1, T1_transform = crop_raster(ortho_file_path, new_plot_leaf['geometry'])
                    nr, nc = T1.shape[1], T1.shape[2]
                    mask_instance = np.zeros((nr, nc))
                    for id in range(len(id_leaf_sel)):
                        single_leaf = shape_leaf.loc[id_leaf_sel[id]]
                        mask_instance[
                            rasterio.features.geometry_mask(
                                [single_leaf['geometry']],
                                out_shape=(nr, nc),
                                transform=T1_transform,
                                invert=True
                            )
                        ] = id_leaf_sel[id]
  
                    # Processing steps here..
                    table_bands = None

                    for b in range(len(bands)):
                        IDX = T1[b]
                        if np.any(np.isinf(np.abs(IDX))):
                            IDX[np.isinf(IDX)] = np.nan
                            IDX = np.log10(IDX)
                            IDX[np.isinf(IDX)] = np.nan
                            IDX = np.reshape(IDX, (nr, nc))

                        IDX = IDX.flatten()

                        if table_bands is None:
                            table_bands = pd.DataFrame(IDX, columns=[bands[b]])
                        else:
                            table_bands = pd.concat([table_bands, pd.Series(IDX, name=bands[b])], axis=1)

                    table_bands = table_bands.dropna()
                    pos = table_bands.index
                    NA_vec = np.ones(nr * nc, dtype=bool)
                    NA_vec[pos] = False

                    raster_data = []

                    first_band = table_bands.iloc[:, 0].values
                    first_band_raster = np.zeros((1, len(pos)))
                    first_band_raster[0, :] = first_band
                    raster_data.append(first_band_raster)

                    for j in range(1, len(bands)):
                        band_data = table_bands.iloc[:, j].values
                        band_raster = np.zeros((1, len(pos)))
                        band_raster[0, :] = band_data
                        raster_data.append(band_raster)

                    raster_data = np.stack(raster_data, axis=0)

                    index_var = pipeline["indices"]
                    table_index = []

                    functions_r = {'MSAVIhyper': MSAVIhy, 'GVIMSS': GVIMSS, 'D678500': D678500, 'NSVDI': NSVDI}
                    for idx in index_var:
                        IDX = functions_r[idx](np.array(bands), raster_data)
                        IDX = np.nan_to_num(IDX, nan=np.log10(IDX))
                        table_index.append(IDX.ravel())

                    table_index = np.array(table_index).T

                    if "angle2cam" in pipeline["coef"] or "angle2sun" in pipeline["coef"]:
                        angle2cam = process_angle_raster(slope_path, new_plot_leaf['geometry'], T1, T1_transform, profile, pos)
                        if np.isnan(angle2cam).all():
                                # Replace all NaN values with 0
                                angle2cam = np.zeros_like(angle2cam)

                    data = np.hstack((table_bands.iloc[:, :5], table_index, angle2cam.reshape(-1, 1)))
                    data = pd.DataFrame(data, columns=pipeline["coef"])

                    ss = np.ones_like(data)
                    ss[np.isnan(data)] = 0

                    pos_non_na = np.where(ss != 1)
                    pos_non_na_row = np.unique(pos_non_na[0])

                    NA_vec_new = np.full(len(data), False)
                    NA_vec_new[pos_non_na_row] = True

                    NA_vec[pos[NA_vec_new]] = True

                    data = data.loc[:, pipeline["coef"]]
                    data = np.array(data)

                    data = data.flatten().astype(float)
                    data = data.reshape(-1, len(pipeline['m']))

                    imputer = SimpleImputer(strategy='mean')
                    data_imputed = imputer.fit_transform(data)

                    pipeline_m = np.array(pipeline['m'])
                    pipeline_s = np.array(pipeline['s'])

                    data_standardized = (data_imputed - pipeline_m) / pipeline_s

                    pred = model.predict(data_standardized)

                    test_pred_grid = np.argmax(pred, axis=-1)

                    non_nan_indices = np.where(NA_vec)
                    mask_pred = np.full((nr * nc), np.nan)

                    for i in range(len(pos)):
                        mask_pred[pos[i]] = test_pred_grid[i]

                    mask_pred = mask_pred.reshape((nr, nc))
                    rraster = np.ones((nr, nc))
                    with rasterio.Env():
                        mask_profile = {
                            'width': T1.shape[2],
                            'height': T1.shape[1],
                            'count': T1.shape[0],
                            'transform': T1_transform,
                            'crs': profile['crs'],
                            'nodata': np.nan,
                            'dtype': 'float32'
                        }
                        with rasterio.open("T1.tif", 'w', **mask_profile) as dst:
                            dst.write(T1)
                    rr = rasterize_shapes(new_plot_leaf,mask_pred.shape,T1_transform)#label_the_leaf_regions(mask_pred, pos, new_plot_leaf)
                    rleaf = rr
                    rr[np.isnan(rr)] = 0
                    rr_ed = rr - erode(rr, F1)

                    seqleaf = np.arange(np.nanmin(rleaf), np.nanmax(rleaf))

                    rleaf_new = np.random.randint(1, 5, size=(rleaf.shape[0], rleaf.shape[1])).astype(str)
                    hex_list = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(new_plot_leaf))]
                    rleaf_new[rleaf_new == np.nan] = '#000000'
                    for idx, val in enumerate(seqleaf):
                        rleaf_new[rleaf == val] = hex_list[idx]
                    rleaf = rleaf_new
                    color_img_in = rleaf
                    instance_vector = color_img_in.flatten()

                    list_hex = hex_list
                    n_ins_test_new = len(list_hex)
                    perform_leaf_df = pd.DataFrame({
                        'id_leaf': range(1, n_ins_test_new + 1),
                        'hex_id': list_hex,
                        'original_leaf_id': new_plot_leaf.index
                    })
                    mask_vec = mask_pred.flatten()

                    per_veg = []

                    for idx, row in perform_leaf_df.iterrows():
                        hex_id = row['hex_id']
                        original_leaf_id = row['original_leaf_id']
                        n_total = np.sum(instance_vector == hex_id)
                        n_diseased = np.sum((mask_vec == class_output.index('diseased') + 1) & (instance_vector == hex_id))
                        n_healthy = np.sum((mask_vec == class_output.index('healthy') + 1) & (instance_vector == hex_id))
                        n_veg = n_diseased + n_healthy
                        per = n_veg * 100 / n_total if n_total > 0 else 0
                        ds = n_diseased * 100 / n_veg if n_veg > 0 else 0
                        per_veg.append([hex_id, n_total, n_diseased, n_healthy, n_veg, per, ds, original_leaf_id])

                    per_veg = pd.DataFrame(per_veg, columns=['hex', 'n_total', 'n_diseased', 'n_healthy', 'n_veg', 'per', 'ds', 'id_leaf'])

                    per_veg = per_veg[per_veg['n_total'] > min_pix_pred]
                    per_veg = per_veg[per_veg['per'] > 20]
                    random.seed(1)
                    n_ins_sample = min(len(per_veg), n_ins)
                    sel_lf = random.sample(range(len(per_veg)), n_ins_sample)
                    sel_hex = per_veg.iloc[sel_lf]['hex'].tolist()
                    per_veg = per_veg.iloc[sel_lf]

                    ds_leaf_sel = per_veg['ds'].mean()
                    mask_vec[~np.isin(instance_vector, per_veg['hex'])] = np.nan
                    instance_vector[~np.isin(instance_vector, sel_hex)] = '#000000'
                    mask_pred = mask_vec.reshape((nr, nc))
                    mask_b = mask_pred

                    if not np.isnan(ds_leaf_sel):
                        condition_border = ds_leaf_sel < th_erosion
                    else:
                        condition_border = True

                    mask_bi = np.zeros((nr, nc))

                    if condition_border:
                        mask_in_t = np.where(~np.isnan(mask_pred), 1, 0)
                        mask_ed_t = erode(mask_in_t, F1)
                        mask_ed_t = erode(mask_ed_t, F1)
                        mask_ed_t = erode(mask_ed_t, F1)
                        mask_ed_t = mask_in_t - mask_ed_t

                        val_edges = mask_pred[mask_ed_t == 1]
                        val_edges[val_edges == class_output.index('diseased') + 1] = class_output.index('extra') + 1
                        mask_pred[mask_ed_t == 1] = val_edges

                    with rasterio.Env():
                        mask_profile = {
                            'width': mask_pred.shape[1],
                            'height': mask_pred.shape[0],
                            'count': 1,
                            'transform': T1_transform,
                            'crs': profile['crs'],
                            'nodata': np.nan,
                            'dtype': 'float32'
                        }
                        with rasterio.open(f"{RESULTS_FOLDER}/leaf_mask.tif", 'w', **mask_profile) as dst:
                            dst.write(mask_pred, 1)
                    cluster_vector = clustering_mask(mask_pred)
                    P1 = process_angle_raster_2(area_path, new_plot_leaf['geometry'], T1,T1_transform,profile)
                    P2 = process_angle_raster_2(slope_path, new_plot_leaf['geometry'], T1,T1_transform,profile)
                    size_vector = P1.flatten()
                    slope_vector = P2.flatten()
                    final_sel_leaf_id = perform_leaf_df[perform_leaf_df['hex_id'].isin(sel_hex)]

                    output_leaf_shape_plot = shape_leaf.loc[final_sel_leaf_id['original_leaf_id'].astype(int)]
                    output_leaf_shape_plot['leaf_id'] = output_leaf_shape_plot.index.astype(str)

                    LEAF_CONDITION = not output_leaf_shape_plot.empty
  
                    if len(output_leaf_shape_plot['leaf_id']) == 0:
                        LEAF_CONDITION = False

                    # Create and return a results dictionary
                    return {
                        'LEAF_CONDITION': LEAF_CONDITION,
                        'mask_pred': mask_pred,
                        'instance_vector': instance_vector,
                        'cluster_vector': cluster_vector,
                        'size_vector': size_vector,
                        'slope_vector': slope_vector,
                        'T1': T1,
                        'n_ins_sample': n_ins_sample,
                        'RESULTS_FOLDER': RESULTS_FOLDER,
                        'folder_name_temp': folder_name_temp,
                        'final_sel_leaf_id': final_sel_leaf_id,
                        'T1_transform': T1_transform,
                        'output_leaf_shape_plot': output_leaf_shape_plot,
                        'rr_ed': rr_ed
                    }

    # Return an empty dictionary if no condition met
    return {
        'LEAF_CONDITION': False,
        'mask_pred': None,
        'instance_vector': None,
        'cluster_vector': None,
        'size_vector': None,
        'slope_vector': None,
        'T1': None,
        'n_ins_sample': 0,
        'RESULTS_FOLDER': '',
        'folder_name_temp': '',
        'final_sel_leaf_id': None,
        'T1_transform': None,
        'output_leaf_shape_plot': None,
        'rr_ed': None
    }


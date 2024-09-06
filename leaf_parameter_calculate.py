from leaf_parameter_utils import *

def leaf_parameter_cal(params):
    # Unpack parameters from the dictionary
    mask_pred = params['leaf_mask_pred']
    leaf_color_matrix = params['leaf_cluster_vector']
    instance_vector = params['leaf_instance_vector']
    size_vector = params['size_vector']
    slope_vector = params['slope_vector']
    n_ins_sample = params['n_ins_sample']
    T1 = params['T1_leaf']
    shape_leaf = params['shape_leaf']
    INSTANCE_FORM = params['INSTANCE_FORM']
    class_output = params['class_output']
    rmux_data_t = params['rmux_data_t']
    rmuy_data_t = params['rmuy_data_t']
    RESULTS_FOLDER = params['RESULTS_FOLDER']
    folder_name_temp = params['folder_name_temp']
    DI_THRESHOLD = params['DI_THRESHOLD']
    plot_id_temp = params['plot_id_temp']
    final_sel_leaf_id = params['final_sel_leaf_id']
    output_leaf_shape_plot = params['output_leaf_shape_plot']

    if (INSTANCE_FORM == "LEAF" or INSTANCE_FORM == "BOTH") and n_ins_sample > 0:
        leaf_prediction = np.array(mask_pred).flatten()

        # Cluster vector
        leaf_cluster_vector = leaf_color_matrix.flatten()

        # Instance vector and number of instances
        leaf_instance_vector = instance_vector
        leaf_n_instances = n_ins_sample

        # Labels
        diseased_label = class_output.index("diseased") + 1
        healthy_label = class_output.index("healthy") + 1

        # Calculate various parameters per leaf instance
        leaf_n_cluster_list = mean_n_cluster_per_instance_plot(
            prediction_vector=leaf_prediction,
            instance_vector=leaf_instance_vector,
            cluster_vector=leaf_cluster_vector,
            info=True
        )
        leaf_n_cluster_list['leaf_id'] = leaf_n_cluster_list['leaf_hex'].map(final_sel_leaf_id.set_index('hex_id')['original_leaf_id'])

        leaf_ds_cover_list = ds_cover_plot(
            diseased_label=diseased_label,
            healthy_label=healthy_label,
            prediction_vector=leaf_prediction,
            instance_vector=leaf_instance_vector,
            cluster_vector=leaf_cluster_vector,
            info=True
        )
        leaf_ds_cover_list['leaf_id'] = leaf_ds_cover_list['leaf_hex'].map(final_sel_leaf_id.set_index('hex_id')['original_leaf_id'])

        leaf_ds_area_list = ds_area_plot(
            diseased_label=diseased_label,
            healthy_label=healthy_label,
            prediction_vector=leaf_prediction,
            instance_vector=leaf_instance_vector,
            cluster_vector=leaf_cluster_vector,
            size_vector=size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=True
        )
        leaf_ds_area_list['leaf_id'] = leaf_ds_area_list['leaf_hex'].map(final_sel_leaf_id.set_index('hex_id')['original_leaf_id'])

        leaf_area_list = veg_area_per_instance(
            diseased_label=diseased_label,
            healthy_label=healthy_label,
            prediction_vector=leaf_prediction,
            instance_vector=leaf_instance_vector,
            cluster_vector=leaf_cluster_vector,
            size_vector=size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=True
        )
        leaf_area_list['leaf_id'] = leaf_area_list['leaf_hex'].map(final_sel_leaf_id.set_index('hex_id')['original_leaf_id'])

        leaf_angle_list = slope_per_instance_per_plot(
            diseased_label=diseased_label,
            healthy_label=healthy_label,
            prediction_vector=leaf_prediction,
            instance_vector=leaf_instance_vector,
            slope_vector=slope_vector,
            info=True
        )
        leaf_angle_list['leaf_id'] = leaf_angle_list['leaf_hex'].map(final_sel_leaf_id.set_index('hex_id')['original_leaf_id'])

        leaf_cluster_area_list = cluster_area_per_plot(
            diseased_label=diseased_label,
            healthy_label=healthy_label,
            prediction_vector=leaf_prediction,
            cluster_vector=leaf_cluster_vector,
            instance_vector=leaf_instance_vector,
            size_vector=size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=True
        )

        # Process cluster area data
        mean_areas = leaf_cluster_area_list.groupby('leaf_hex')['data_instance'].mean().reset_index()
        mean_areas.rename(columns={'data_instance': 'mean_area'}, inplace=True)
        leaf_cluster_area_list = leaf_angle_list.merge(mean_areas, on='leaf_hex', how='left')
        leaf_cluster_area_list['data_instance'] = leaf_cluster_area_list['mean_area']
        leaf_cluster_area_list.drop(columns=['mean_area'], inplace=True)

        # Prepare final DataFrame of leaf instance parameters
        leaf_instance_parameter_list = pd.DataFrame({
            'leaf_id': leaf_cluster_area_list['leaf_id'].astype(int),
            'n_cluster_param': leaf_n_cluster_list['data_instance'],
            'ds_cover_param': leaf_ds_cover_list['data_instance'],
            'ds_area_param': leaf_ds_area_list['data_instance'],
            'area_param': leaf_area_list['data_instance'],
            'angle_param': leaf_angle_list['data_instance'],
            'cluster_area_param': leaf_cluster_area_list['data_instance']
        })

        # Merge with output leaf shape plot
        output_leaf_shape_plot['leaf_id'] = output_leaf_shape_plot['leaf_id'].astype(int)
        output_leaf_shape_plot = output_leaf_shape_plot.merge(leaf_instance_parameter_list, on='leaf_id', how='left')

        # Adjust leaf ID for file output
        output_leaf_shape_plot['leaf_id'] = output_leaf_shape_plot['leaf_id'].astype(int) - 1

        # Save results
        output_leaf_shape_plot.to_file(
            driver='GPKG',
            filename=f"{RESULTS_FOLDER}/leaf_parameter_{folder_name_temp}.gpkg",
            layer=f"leaf_parameter_{folder_name_temp}",
            overwrite=True
        )

        leaf_instance_parameter_list.to_csv(f"{RESULTS_FOLDER}/leaf_parameter.csv", index=False)

        # Plot-level parameters
        leaf_di = di_tv_plot(leaf_prediction, leaf_instance_vector, leaf_cluster_vector, tv=5)
        leaf_mean_n_cluster = mean_n_cluster_per_instance_plot(leaf_prediction, leaf_instance_vector, leaf_cluster_vector, info=False)
        leaf_ds_cover = ds_cover_plot(diseased_label, healthy_label, leaf_prediction, leaf_instance_vector, leaf_cluster_vector, info=False)
        leaf_ds_area = ds_area_plot(diseased_label, healthy_label, leaf_prediction, leaf_instance_vector, leaf_cluster_vector, size_vector, rmux_data_t, rmuy_data_t, info=False)
        leaf_area = veg_area_per_instance(diseased_label, healthy_label, leaf_prediction, leaf_instance_vector, leaf_cluster_vector, size_vector, rmux_data_t, rmuy_data_t, False)
        leaf_angle = slope_per_instance_per_plot(diseased_label, healthy_label, leaf_prediction, leaf_instance_vector, slope_vector, info=False)
        leaf_cluster_area = cluster_area_per_plot(diseased_label, healthy_label, leaf_prediction, leaf_instance_vector, leaf_cluster_vector, size_vector, rmux_data_t, rmuy_data_t, info=False)

        condition_application = leaf_di > DI_THRESHOLD * 100
        DECISION_PLOT_temp = "APPLICATION" if condition_application else "HEALTHY"
    else:
        leaf_mean_n_cluster = np.nan
        leaf_di = np.nan
        leaf_ds_cover = np.nan
        leaf_ds_area = np.nan
        leaf_area = np.nan
        leaf_angle = np.nan
        leaf_cluster_area = np.nan
        DECISION_PLOT_temp = np.nan

    # Prepare the results dictionary
    leaf_parameters_t = {
        'plot_shape': plot_id_temp,
        'leaf_mean_n_cluster': leaf_mean_n_cluster,
        'leaf_di': leaf_di,
        'leaf_ds_cover': leaf_ds_cover,
        'leaf_ds_area': leaf_ds_area,
        'leaf_area': leaf_area,
        'leaf_angle': leaf_angle,
        'leaf_cluster_area': leaf_cluster_area,
        'DECISION_PLOT_temp':DECISION_PLOT_temp
    }

    return {'DECISION_PLOT_temp':DECISION_PLOT_temp, 'leaf_parameter':leaf_parameters_t}
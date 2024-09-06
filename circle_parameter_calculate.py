from leaf_parameter_utils import *
import os
import numpy as np
import pandas as pd
import geopandas as gpd

def circle_parameter_cal(params):
    # Unpack parameters from the dictionary
    INSTANCE_FORM = params['INSTANCE_FORM']
    circle_prediction = params['circle_prediction']
    circle_cluster_vector = params['circle_cluster_vector']
    circle_instance_vector = params['circle_instance_vector']
    class_output = params['class_output']
    circle_size_vector = params['circle_size_vector']
    rmux_data_t = params['rmux_data_t']
    rmuy_data_t = params['rmuy_data_t']
    RESULTS_FOLDER = params['RESULTS_FOLDER']
    plot_id_temp = params['plot_id_temp']
    folder_name_temp = params['folder_name_temp']
    output_circle_shape_plot = params['output_circle_shape_plot']

    if INSTANCE_FORM == "CIRCLE" or INSTANCE_FORM == "BOTH":
        # Calculate n cluster
        circle_n_cluster_list = mean_n_cluster_per_instance_plot(
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            info=True
        )
        circle_n_cluster_list.columns.values[1] = "circle_id"
        circle_n_cluster_list = circle_n_cluster_list.sort_values("circle_id").reset_index(drop=True)

        # Calculate ds cover
        circle_ds_cover_list = ds_cover_plot(
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            info=True
        )
        circle_ds_cover_list.columns.values[1] = "circle_id"
        circle_ds_cover_list = circle_ds_cover_list.sort_values("circle_id").reset_index(drop=True)

        # Calculate ds area
        circle_ds_area_list = ds_area_plot(
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            size_vector=circle_size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=True
        )
        circle_ds_area_list.columns.values[1] = "circle_id"
        circle_ds_area_list = circle_ds_area_list.sort_values("circle_id").reset_index(drop=True)

        # Calculate healthy and diseased cover
        circle_hc_list = healthy_cover(
            soil_label=class_output.index("soil") + 1,
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            info=True
        )
        circle_hc_list.columns.values[1] = "circle_id"
        circle_hc_list = circle_hc_list.sort_values("circle_id").reset_index(drop=True)

        circle_dc_list = diseased_cover(
            soil_label=class_output.index("soil") + 1,
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            info=True
        )
        circle_dc_list.columns.values[1] = "circle_id"
        circle_dc_list = circle_dc_list.sort_values("circle_id").reset_index(drop=True)

        # Calculate plot cover
        circle_pc_list = pc_plot(
            soil_label=class_output.index("soil") + 1,
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            info=True
        )
        circle_pc_list.columns.values[1] = "circle_id"
        circle_pc_list = circle_pc_list.sort_values("circle_id").reset_index(drop=True)

        # Calculate area parameters
        circle_diseased_area_list = diseased_area_per_instance(
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            size_vector=circle_size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=True
        )
        circle_diseased_area_list.columns.values[1] = "circle_id"
        circle_diseased_area_list = circle_diseased_area_list.sort_values("circle_id").reset_index(drop=True)

        circle_healthy_area_list = healthy_area_per_instance(
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            size_vector=circle_size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=True
        )
        circle_healthy_area_list.columns.values[1] = "circle_id"
        circle_healthy_area_list = circle_healthy_area_list.sort_values("circle_id").reset_index(drop=True)

        circle_veg_area_list = veg_area_per_instance(
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            size_vector=circle_size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=True
        )
        circle_veg_area_list.columns.values[1] = "circle_id"
        circle_veg_area_list = circle_veg_area_list.sort_values("circle_id").reset_index(drop=True)

        # Summary instance parameters
        circle_instance_parameter_list = pd.DataFrame({
            'circle_id': circle_veg_area_list['circle_id'],
            'n_cluster': circle_n_cluster_list['data_instance'],
            'ds_cover': circle_ds_cover_list['data_instance'],
            'ds_area': circle_ds_area_list['data_instance'],
            'diseased_area': circle_diseased_area_list['data_instance'],
            'healthy_area': circle_healthy_area_list['data_instance'],
            'veg_area': circle_veg_area_list['data_instance'],
            'hc': circle_hc_list['data_instance'],
            'dc': circle_dc_list['data_instance'],
            'pc': circle_pc_list['data_instance']
        })

        # Print GPKG
        output_circle_shape_plot = gpd.GeoDataFrame(output_circle_shape_plot)
        output_circle_shape_plot['circle_id'] = output_circle_shape_plot['circle_id'].astype(str)
        circle_instance_parameter_list['circle_id'] = circle_instance_parameter_list['circle_id'].astype(str)

        output_circle_shape_plot = output_circle_shape_plot.merge(circle_instance_parameter_list, on='circle_id', how='left')

        output_circle_shape_plot.to_file(
            filename=os.path.join(RESULTS_FOLDER, f'circle_parameter_{folder_name_temp}.gpkg'),
            driver='GPKG'
        )

        # Save CSV
        circle_instance_parameter_list.to_csv(
            os.path.join(RESULTS_FOLDER, "circle_parameter.csv"),
            index=False
        )

        # Parameters PLOT level
        circle_mean_n_cluster = mean_n_cluster_per_instance_plot(
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            info=False
        )
        circle_di = di_plot(
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector
        )
        circle_ds_cover = ds_cover_plot(
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            info=False
        )
        circle_ds_area = ds_area_plot(
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            size_vector=circle_size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=False
        )
        circle_hc = healthy_cover(
            soil_label=class_output.index("soil") + 1,
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            info=False
        )
        circle_dc = diseased_cover(
            soil_label=class_output.index("soil") + 1,
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            info=False
        )
        circle_pc = pc_plot(
            soil_label=class_output.index("soil") + 1,
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            info=False
        )
        circle_diseased_area = diseased_area_per_instance(
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            size_vector=circle_size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=False
        )
        circle_healthy_area = healthy_area_per_instance(
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            size_vector=circle_size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=False
        )
        circle_veg_area = veg_area_per_instance(
            diseased_label=class_output.index("diseased") + 1,
            healthy_label=class_output.index("healthy") + 1,
            prediction_vector=circle_prediction,
            instance_vector=circle_instance_vector,
            cluster_vector=circle_cluster_vector,
            size_vector=circle_size_vector,
            x_res=rmux_data_t,
            y_res=rmuy_data_t,
            info=False
        )
    else:
        circle_mean_n_cluster = np.nan
        circle_di = np.nan
        circle_ds_cover = np.nan
        circle_ds_area = np.nan
        circle_hc = np.nan
        circle_dc = np.nan
        circle_pc = np.nan
        circle_diseased_area = np.nan
        circle_healthy_area = np.nan
        circle_veg_area = np.nan

    circle_parameters_t = {
        'plot_shape': plot_id_temp,
        'circle_mean_n_cluster': circle_mean_n_cluster,
        'circle_di': circle_di,
        'circle_ds_cover': circle_ds_cover,
        'circle_ds_area': circle_ds_area,
        'circle_hc': circle_hc,
        'circle_dc': circle_dc,
        'circle_pc': circle_pc,
        'circle_diseased_area': circle_diseased_area,
        'circle_healthy_area': circle_healthy_area,
        'circle_veg_area': circle_veg_area
    }

    return {'circle_parameter':circle_parameters_t}

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import pandas as pd


def mean_n_cluster_per_instance_plot(prediction_vector, instance_vector, cluster_vector, info):
    # Create a DataFrame
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector
    })

    # Remove background hex color instances
    data = data[data['instance'] != '#000000']
    
    # Replace '#000000' in cluster with NaN
    data.loc[data['cluster'] == '#000000', 'cluster'] = np.nan

    # Get unique instances
    unique_instances = data['instance'].unique()
    n_instances = len(unique_instances)
    
    data_instance = []
    leaf_hex_instance = []

    for instance in unique_instances:
        data_t = data[data['instance'] == instance]
        n_cluster_temp = len(data_t['cluster'].dropna().unique())

        data_instance.append(n_cluster_temp)
        leaf_hex_instance.append(instance)
    
    result_df = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })
    
    if not info:
        # Calculate mean if info is False
        result = result_df['data_instance'].mean()
    else:
        # Return the DataFrame if info is True
        result = result_df
    
    return result








def ds_cover_plot(diseased_label, healthy_label, prediction_vector, instance_vector, cluster_vector, info):
    # Create a DataFrame
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector
    })

    # Remove background hex color instances
    data = data[data['instance'] != '#000000']
    
    # Replace '#000000' in cluster with NaN
    data.loc[data['cluster'] == '#000000', 'cluster'] = np.nan

    # Get unique instances
    unique_instances = data['instance'].unique()
    
    data_instance = []
    leaf_hex_instance = []

    for instance in unique_instances:
        data_t = data[data['instance'] == instance]
        # Calculate percentage of diseased labels
        diseased_count = (data_t['prediction'] == diseased_label).sum()
        total_count = ((data_t['prediction'] == diseased_label) | (data_t['prediction'] == healthy_label)).sum()
        ds_t = (diseased_count * 100.0 / total_count) if total_count > 0 else np.nan

        data_instance.append(ds_t)
        leaf_hex_instance.append(instance)
    
    result_df = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })
    
    if not info:
        # Calculate mean if info is False
        result = result_df['data_instance'].mean()
    else:
        # Return the DataFrame if info is True
        result = result_df
    
    return result




def ds_area_plot(diseased_label, healthy_label, prediction_vector, instance_vector, cluster_vector, size_vector, x_res, y_res, info):
    # Create a DataFrame
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector,
        'area_coef': size_vector,
        'x_res': x_res,
        'y_res': y_res
    })

    # Remove background hex color instances
    data = data[data['instance'] != '#000000']
    
    # Replace '#000000' in cluster with NaN
    data.loc[data['cluster'] == '#000000', 'cluster'] = np.nan

    # Get unique instances
    unique_instances = data['instance'].unique()
    
    data_instance = []
    leaf_hex_instance = []

    for instance in unique_instances:
        data_t = data[data['instance'] == instance]
        
        # Calculate areas
        area_diseased_cm2 = (data_t.loc[data_t['prediction'] == diseased_label, 'area_coef'] *
                             data_t.loc[data_t['prediction'] == diseased_label, 'x_res'] *
                             data_t.loc[data_t['prediction'] == diseased_label, 'y_res']).sum() * 100 * 100
        
        area_healthy_cm2 = (data_t.loc[data_t['prediction'] == healthy_label, 'area_coef'] *
                            data_t.loc[data_t['prediction'] == healthy_label, 'x_res'] *
                            data_t.loc[data_t['prediction'] == healthy_label, 'y_res']).sum() * 100 * 100
        
        area_vegetation_cm2 = area_diseased_cm2 + area_healthy_cm2
        
        # Calculate percentage of diseased area
        ds_area = (area_diseased_cm2 * 100 / area_vegetation_cm2) if area_vegetation_cm2 > 0 else np.nan
        
        data_instance.append(ds_area)
        leaf_hex_instance.append(instance)
    
    result_df = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })
    
    if not info:
        # Calculate mean if info is False
        result = result_df['data_instance'].mean()
    else:
        # Return the DataFrame if info is True
        result = result_df
    
    return result



    


def veg_area_per_instance(diseased_label, healthy_label, prediction_vector, instance_vector, cluster_vector, size_vector, x_res, y_res, info):
    # Create a DataFrame
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector,
        'area_coef': size_vector,
        'x_res': x_res,
        'y_res': y_res
    })

    # Remove background hex color instances
    data = data[data['instance'] != '#000000']
    
    # Replace '#000000' in cluster with NaN
    data.loc[data['cluster'] == '#000000', 'cluster'] = np.nan

    # Get unique instances
    unique_instances = data['instance'].unique()
    
    data_instance = []
    leaf_hex_instance = []

    for instance in unique_instances:
        data_t = data[data['instance'] == instance]
        
        # Calculate areas
        area_diseased_cm2 = (data_t.loc[data_t['prediction'] == diseased_label, 'area_coef'] *
                             data_t.loc[data_t['prediction'] == diseased_label, 'x_res'] *
                             data_t.loc[data_t['prediction'] == diseased_label, 'y_res']).sum() * 100 * 100
        
        area_healthy_cm2 = (data_t.loc[data_t['prediction'] == healthy_label, 'area_coef'] *
                            data_t.loc[data_t['prediction'] == healthy_label, 'x_res'] *
                            data_t.loc[data_t['prediction'] == healthy_label, 'y_res']).sum() * 100 * 100
        
        area_vegetation_cm2 = area_diseased_cm2 + area_healthy_cm2
        
        data_instance.append(area_vegetation_cm2)
        leaf_hex_instance.append(instance)
    
    result_df = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })
    
    if not info:
        # Calculate mean if info is False
        result = result_df['data_instance'].mean()
    else:
        # Return the DataFrame if info is True
        result = result_df
    
    return result




def slope_per_instance_per_plot(diseased_label, healthy_label, prediction_vector, instance_vector, slope_vector, info):
    # Create a DataFrame
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'slope': slope_vector
    })

    # Remove background hex color instances
    data = data[data['instance'] != '#000000']

    # Get unique instances
    unique_instances = data['instance'].unique()
    
    data_instance = []
    leaf_hex_instance = []

    for instance in unique_instances:
        data_t = data[data['instance'] == instance]
        
        # Calculate mean slope for diseased and healthy labels
        relevant_slope = data_t.loc[data_t['prediction'].isin([diseased_label, healthy_label]), 'slope']
        mean_slope = relevant_slope.mean() if not relevant_slope.empty else np.nan
        
        data_instance.append(mean_slope)
        leaf_hex_instance.append(instance)
    
    result_df = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })
    
    if not info:
        # Calculate mean if info is False
        result = result_df['data_instance'].mean()
    else:
        # Return the DataFrame if info is True
        result = result_df
    
    return result


def cluster_area_per_plot(diseased_label, healthy_label, prediction_vector, cluster_vector, instance_vector, size_vector, x_res, y_res, info):
    # Create a DataFrame
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector,
        'area_coef': size_vector,
        'x_res': x_res,
        'y_res': y_res
    })

    # Replace '#000000' in cluster with NaN
    data.loc[data['cluster'] == '#000000', 'cluster'] = np.nan
    
    # Get unique clusters
    unique_clusters = data['cluster'].dropna().unique()
    
    data_instance = []
    leaf_hex_instance = []

    for cluster in unique_clusters:
        data_t = data[data['cluster'] == cluster]
        
        # Calculate area of diseased labels
        area_diseased_cm2 = (data_t.loc[data_t['prediction'] == diseased_label, 'area_coef'] *
                             data_t.loc[data_t['prediction'] == diseased_label, 'x_res'] *
                             data_t.loc[data_t['prediction'] == diseased_label, 'y_res']).sum() * 100 * 100
        
        data_instance.append(area_diseased_cm2)
        
        # Get instance value associated with this cluster
        instance_value = data_t['instance'].dropna().iloc[0] if not data_t['instance'].dropna().empty else np.nan
        leaf_hex_instance.append(instance_value)
    
    result_df = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })
    
    if not info:
        # Calculate mean if info is False
        result = result_df['data_instance'].mean()
    else:
        # Return the DataFrame if info is True
        result = result_df
    
    return result








def di_tv_plot(prediction_vector, instance_vector, cluster_vector, tv):
    # Create DataFrame
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector
    })
    
    # Remove background color
    data = data[data['instance'] != '#000000']
    
    # Replace background spots with NaN
    data.loc[data['cluster'] == '#000000', 'cluster'] = np.nan
    
    # Number of unique instances
    unique_instances = data['instance'].unique()
    
    # Calculate number of clusters per instance
    clusters_per_instance = []
    for instance in unique_instances:
        data_t = data[data['instance'] == instance]
        n_clusters = data_t['cluster'].nunique()
        clusters_per_instance.append(n_clusters)
    
    # Calculate DI
    di = (np.sum(np.array(clusters_per_instance) >= tv) * 100) / len(unique_instances)
    
    return di



def healthy_cover(diseased_label, healthy_label, soil_label, prediction_vector, instance_vector, cluster_vector, info):
    # Create a DataFrame from the input vectors
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector
    })
    
    # Remove rows where instance is '#000000'
    data = data[data['instance'] != '#000000']
    
    # Replace '#000000' in cluster column with NaN
    data['cluster'].replace('#000000', np.nan, inplace=True)
    
    # Get the unique instances
    unique_instances = data['instance'].unique()
    
    data_instance = []
    leaf_hex_instance = []
    
    # Loop through each unique instance
    for instance in unique_instances:
        data_t = data[data['instance'] == instance]
        
        healthy_count = (data_t['prediction'] == healthy_label).sum()
        total_count = ((data_t['prediction'] == diseased_label) | 
                       (data_t['prediction'] == healthy_label) | 
                       (data_t['prediction'] == soil_label)).sum()
        
        if total_count > 0:
            hc_t = healthy_count * 100 / total_count
        else:
            hc_t = np.nan
        
        data_instance.append(hc_t)
        
        # Get the first non-NA instance for the current cluster
        first_instance = data_t['instance'].dropna().iloc[0] if not data_t['instance'].dropna().empty else np.nan
        leaf_hex_instance.append(first_instance)
    
    result = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })
    
    if not info:
        return result['data_instance'].mean()
    else:
        return result






def diseased_cover(diseased_label, healthy_label, soil_label, prediction_vector, instance_vector, cluster_vector, info):
    # Create a DataFrame from the input vectors
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector
    })
    
    # Remove rows where instance is '#000000'
    data = data[data['instance'] != '#000000']
    
    # Replace '#000000' in cluster column with NaN
    data['cluster'].replace('#000000', np.nan, inplace=True)
    
    # Get the unique instances
    unique_instances = data['instance'].unique()
    
    data_instance = []
    leaf_hex_instance = []
    
    # Loop through each unique instance
    for instance in unique_instances:
        data_t = data[data['instance'] == instance]
        
        diseased_count = (data_t['prediction'] == diseased_label).sum()
        total_count = ((data_t['prediction'] == diseased_label) | 
                       (data_t['prediction'] == healthy_label) | 
                       (data_t['prediction'] == soil_label)).sum()
        
        if total_count > 0:
            dc_t = diseased_count * 100 / total_count
        else:
            dc_t = np.nan
        
        data_instance.append(dc_t)
        
        # Get the first non-NA instance for the current cluster
        first_instance = data_t['instance'].dropna().iloc[0] if not data_t['instance'].dropna().empty else np.nan
        leaf_hex_instance.append(first_instance)
    
    result = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })
    
    if not info:
        return result['data_instance'].mean()
    else:
        return result



def di_plot(prediction_vector, instance_vector, cluster_vector):
    # Merge data
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector
    })

    # Filter out background instances
    data = data[data['instance'] != "#000000"]

    # Filter out background clusters
    data.loc[data['cluster'] == "#000000", 'cluster'] = None

    # Number of instances
    length_instances = len(data['instance'].unique())

    # Initialize list to store diversity index per instance
    data_instance = []

    # Calculate diversity index for each instance
    for instance_id in data['instance'].unique():
        data_t = data[data['instance'] == instance_id]
        n_cluster_temp = len(data_t['cluster'].dropna().unique())
        data_instance.append(n_cluster_temp)

    # Compute diversity index (DI)
    di = len([x for x in data_instance if x != 0]) * 100 / length_instances

    return di


def pc_plot(diseased_label, healthy_label, soil_label, prediction_vector, instance_vector, cluster_vector, info):
    # Create a DataFrame from the input vectors
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector
    })
    
    # Remove rows where instance is '#000000'
    data = data[data['instance'] != '#000000']
    
    # Replace '#000000' in cluster column with NaN
    data['cluster'].replace('#000000', np.nan, inplace=True)
    
    # Get the unique instances
    unique_instances = data['instance'].unique()
    
    data_instance = []
    leaf_hex_instance = []
    
    # Loop through each unique instance
    for instance in unique_instances:
        data_t = data[data['instance'] == instance]
        
        veg_count = (data_t['prediction'] == diseased_label).sum() + (data_t['prediction'] == healthy_label).sum()
        total_count = ((data_t['prediction'] == diseased_label) | 
                       (data_t['prediction'] == healthy_label) | 
                       (data_t['prediction'] == soil_label)).sum()
        
        if total_count > 0:
            pc_t = veg_count * 100 / total_count
        else:
            pc_t = np.nan
        
        data_instance.append(pc_t)
        
        # Get the first non-NA instance for the current cluster
        first_instance = data_t['instance'].dropna().iloc[0] if not data_t['instance'].dropna().empty else np.nan
        leaf_hex_instance.append(first_instance)
    
    result = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })
    
    if not info:
        return result['data_instance'].mean()
    else:
        return result



def diseased_area_per_instance(diseased_label, healthy_label, prediction_vector, instance_vector, cluster_vector, size_vector, x_res, y_res, info):
    # Create DataFrame from input vectors
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector,
        'area_coef': size_vector,
        'x_res': x_res,
        'y_res': y_res
    })

    # Remove background hex color (assuming background hex color is '#000000')
    data = data[data['instance'] != '#000000']

    # Remove spots background (set cluster to NaN where cluster is '#000000')
    data.loc[data['cluster'] == '#000000', 'cluster'] = np.nan

    # Compute length of instances and their hex codes
    level_instances = data['instance'].unique()
    length_instances = len(level_instances)

    data_instance = []
    leaf_hex_instance = []

    # Loop through each instance level
    for instance in level_instances:
        data_t = data[data['instance'] == instance]

        # Compute area of diseased instances
        area_diseased_cm2 = (data_t.loc[data_t['prediction'] == diseased_label, 'area_coef'] * 
                             data_t.loc[data_t['prediction'] == diseased_label, 'x_res'] *
                             data_t.loc[data_t['prediction'] == diseased_label, 'y_res']).sum() * 100 * 100

        data_instance.append(area_diseased_cm2)
        leaf_hex_instance.append(data_t['instance'].dropna().iloc[0])

    # Create DataFrame with results
    result_df = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })

    if not info:
        return result_df['data_instance'].mean()
    
    return result_df


def healthy_area_per_instance(diseased_label, healthy_label, prediction_vector, instance_vector, cluster_vector, size_vector, x_res, y_res, info):
    # Create DataFrame from input vectors
    data = pd.DataFrame({
        'prediction': prediction_vector,
        'instance': instance_vector,
        'cluster': cluster_vector,
        'area_coef': size_vector,
        'x_res': x_res,
        'y_res': y_res
    })

    # Remove background hex color (assuming background hex color is '#000000')
    data = data[data['instance'] != '#000000']

    # Remove spots background (set cluster to NaN where cluster is '#000000')
    data.loc[data['cluster'] == '#000000', 'cluster'] = np.nan

    # Compute length of instances and their hex codes
    level_instances = data['instance'].unique()
    length_instances = len(level_instances)

    data_instance = []
    leaf_hex_instance = []

    # Loop through each instance level
    for instance in level_instances:
        data_t = data[data['instance'] == instance]

        # Compute area of healthy instances
        area_healthy_cm2 = (data_t.loc[data_t['prediction'] == healthy_label, 'area_coef'] * 
                            data_t.loc[data_t['prediction'] == healthy_label, 'x_res'] *
                            data_t.loc[data_t['prediction'] == healthy_label, 'y_res']).sum() * 100 * 100

        data_instance.append(area_healthy_cm2)
        leaf_hex_instance.append(data_t['instance'].dropna().iloc[0])

    # Create DataFrame with results
    result_df = pd.DataFrame({
        'data_instance': data_instance,
        'leaf_hex': leaf_hex_instance
    })

    if not info:
        return result_df['data_instance'].mean()
    
    return result_df

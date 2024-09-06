from leaf_utils import *


def imaging_leaf_updated(T1_leaf,RESULTS_FOLDER,rr_ed,leaf_cluster_vector,leaf_mask_pred,shape_temp,ortho_file_path):
    Im_RGB = get_cropped_regio(ortho_file_path,shape_temp)
    Im_RGB = np.array(Im_RGB)
        # File paths (replace with actual paths)
    IMAGING_PATH = RESULTS_FOLDER

    # Save the base RGB image
    file_name_t = f"{IMAGING_PATH}/LEAF-RGB_roi.png"
    save_image(Im_RGB, file_name_t)

    

    array1 = rr_ed # Example smaller array
    array2 = Im_RGB  # Example larger array

    # Pad array1 to match the shape of array2
    rr_ed = resize_array(array1, array2)


    array1 = leaf_cluster_vector # Example smaller array
    array2 = Im_RGB  # Example larger array

    # Pad array1 to match the shape of array2
    leaf_cluster_vector = resize_array(array1, array2)


    array1 = leaf_mask_pred # Example smaller array
    array2 = Im_RGB  # Example larger array

    # Pad array1 to match the shape of array2
    leaf_mask_pred = resize_array(array1, array2)

    # For instance, assuming rr_ed and color_img_ed are numpy arrays
    # You would define these arrays based on your actual data
    # rr_ed = rr_ed # Example placeholder
    labels = find_unique_labels(rr_ed)
    color_map = {label_r: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label_r in labels}
    # rr_ed[50:60, 50:60] = 1  # Example region

    Col_Mat = apply_colormap(Im_RGB, rr_ed, color_map)
    file_name_t = f"{IMAGING_PATH}/LEAF-RGB_instance.png"
    save_image(Col_Mat, file_name_t)

    # Similarly handle color_img_spot and mask with proper logic
    # For demonstration, placeholders are used
    color_img_spot = leaf_cluster_vector
    Col_Mat = colormap(Im_RGB, color_img_spot)
    file_name_t = f"{IMAGING_PATH}/LEAF-RGB_clusters.png"
    save_image(Col_Mat, file_name_t)

    # Prediction example (assuming mask and class_output are properly defined)

    class_dict = {1: "#FFFFFF", 3: "#00FF00", 2: "#FF0000", 4: "#000000"}
    Col_Mat = apply_colormap(Im_RGB, leaf_mask_pred, class_dict)
    file_name_t = f"{IMAGING_PATH}/LEAF-RGB_prediction.png"
    save_image(Col_Mat, file_name_t)
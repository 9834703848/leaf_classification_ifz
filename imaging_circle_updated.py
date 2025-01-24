from leaf_utils import *


def imaging_circle_updated(ortho_file_path,shape_temp,RESULTS_FOLDER,rr_ed_circle,circle_cluster_vector,circle_prediction):
    # Replace with actual I1 data
    
    Im_RGB = get_cropped_regio(ortho_file_path,shape_temp)
    Im_RGB = np.array(Im_RGB)
   
    array1 = rr_ed_circle # Example smaller array
    array2 = Im_RGB  # Example larger array

    # Pad array1 to match the shape of array2
    rr_ed_circle_padded = resize_array(array1, array2)


    array1 = circle_cluster_vector # Example smaller array
    array2 = Im_RGB  # Example larger array

    # Pad array1 to match the shape of array2
    circle_cluster_vector_padded = resize_array(array1, array2)


    array1 = circle_prediction # Example smaller array
    array2 = Im_RGB  # Example larger array

    # Pad array1 to match the shape of array2
    circle_prediction_padded = resize_array(array1, array2)
        
        # File paths (replace with actual paths)
    IMAGING_PATH = RESULTS_FOLDER

    # Save the base RGB image
    file_name_t = f"{IMAGING_PATH}/CIRCLE-RGB_roi.png"
    save_image(Im_RGB, file_name_t)

    # For instance, assuming rr_ed and color_img_ed are numpy arrays
    # You would define these arrays based on your actual data
    # rr_ed = rr_ed # Example placeholder
    labels = find_unique_labels(rr_ed_circle_padded)
    color_map = {label_r: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label_r in labels}
    # rr_ed[50:60, 50:60] = 1  # Example region

    Col_Mat = apply_colormap(Im_RGB, rr_ed_circle_padded, color_map)
    file_name_t = f"{IMAGING_PATH}/CIRCLE-RGB_instance.png"
    save_image(Col_Mat, file_name_t)

    # Similarly handle color_img_spot and mask with proper logic
    # For demonstration, placeholders are used
    color_img_spot = circle_cluster_vector_padded
    Col_Mat = colormap(Im_RGB, color_img_spot)
    file_name_t = f"{IMAGING_PATH}/CIRCLE-RGB_clusters.png"
    save_image(Col_Mat, file_name_t)

    # Prediction example (assuming mask and class_output are properly defined)

    class_dict = {1: "#FFFFFF", 2: "#00FF00", 3: "#FF0000", 4: "#000000"}
    Col_Mat = apply_colormap(Im_RGB, circle_prediction_padded, class_dict)
    file_name_t = f"{IMAGING_PATH}/CIRCLE-RGB_prediction.png"
    save_image(Col_Mat, file_name_t)
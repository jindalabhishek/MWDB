


def get_output_file_path(task_id,feature_model,topic,reduction_technique):
    return "_".join([str(task_id),str(feature_model),str(reduction_technique),str(topic)])+".json"
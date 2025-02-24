def evaluate_spatial_relationships(model, objects, relations, image_path):
    """ Evaluate spatial relationships for an image based on detected objects. """
    detection_result = model.detect_objects(image_path, objects)
    bboxes = [detection_result[obj]['bbox'] for obj in objects if obj in detection_result]
    
    success_count = 0
    for (obj_a, obj_b, expected_relation) in zip(objects[:-1], objects[1:], relations):
        relation = determine_relative_position(bboxes[objects.index(obj_a)], bboxes[objects.index(obj_b)])
        if relation == expected_relation:
            success_count += 1

    accuracy = success_count / len(relations) if relations else 0
    return {
        'img_id': image_path.split('/')[-1].split('.')[0],
        'detected_relations': relations,
        'accuracy': accuracy
    }
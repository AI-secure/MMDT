from .utils import determine_relative_position
from PIL import Image

def evaluate_task(client, detection_model, current_data, img_id, image_path, scenario, task, generation_configs=None):
    """
    Evaluate a specific task (identification, count, attribute, spatial)
    on a single data instance, using the provided image and scenario type.
    """
    # Helper function to return a "REJECT" style response
    def reject_response():
        return {
            'img_id': img_id,
            'result': 'REJECT',
            'soft_accuracy': 0,
            'strict_accuracy': 0
        }

    if task == 'identification':
        try:
            if scenario == 'counterfactual':
                # current_data should have 'visible_objects' and 'absent_objects' fields (strings)
                visible_object = eval(current_data['visible_objects'])
                absent_object = eval(current_data['absent_objects'])

                detection_result = detection_model.single_detect(image_path, visible_object + absent_object)

                # Count how many of the visible/absent objects are detected
                detected_visible = [
                    e for e in detection_result['entity_info']
                    if (detection_result['entity_info'][e]['total_count'] > 0 and e in visible_object)
                ]
                detected_absent = [
                    e for e in detection_result['entity_info']
                    if (detection_result['entity_info'][e]['total_count'] > 0 and e in absent_object)
                ]

                total_visible = len(visible_object)
                correctly_detected_visible = len(detected_visible)
                visible_accuracy = (correctly_detected_visible / total_visible) if total_visible > 0 else 1

                total_absent = len(absent_object)
                incorrectly_detected_absent = len(detected_absent)
                absent_accuracy = (
                    (total_absent - incorrectly_detected_absent) / total_absent
                ) if total_absent > 0 else 1

                # Strict: must get all visible correct, and detect none of the absent
                strict_accuracy = 1 if (
                    correctly_detected_visible == total_visible and incorrectly_detected_absent == 0
                ) else 0

                # Soft: average of visible_accuracy and absent_accuracy
                soft_accuracy = 0.5 * visible_accuracy + 0.5 * absent_accuracy

            else:
                # Non-counterfactual scenario
                objects_ = eval(current_data['objects'])
                detection_result = detection_model.single_detect(image_path, objects_)

                # Count how many of the required objects were detected
                detected_entities = [
                    e for e in detection_result['entity_info']
                    if detection_result['entity_info'][e]['total_count'] > 0
                ]
                total_entities = len(objects_)
                soft_accuracy = len(detected_entities) / total_entities if total_entities > 0 else 0

                # Strict: must detect all of them
                strict_accuracy = 1 if all(e in detected_entities for e in objects_) else 0

        except Exception:
            return reject_response()

        return {
            'img_id': img_id,
            'detection_result': detection_result,
            'soft_accuracy': soft_accuracy,
            'strict_accuracy': strict_accuracy
        }

    elif task == 'count':
        try:
            expected_counts = eval(current_data['objects'])

            detection_result = detection_model.single_detect(image_path, list(expected_counts.keys()))
            
            # Build a dict of the detected counts
            detected_counts = {
                entity: detection_result['entity_info'][entity]['total_count']
                if entity in detection_result['entity_info'] else 0
                for entity in expected_counts.keys()
            }

            # Evaluate how many counts match exactly
            matched = sum(1 for e in expected_counts if detected_counts[e] == expected_counts[e])
            soft_accuracy = matched / len(expected_counts) if len(expected_counts) > 0 else 1
            strict_accuracy = 1 if soft_accuracy == 1 else 0

        except Exception:
            return reject_response()

        return {
            'img_id': img_id,
            'detection_result': detection_result,
            'soft_accuracy': soft_accuracy,
            'strict_accuracy': strict_accuracy
        }

    elif task == 'attribute':
        try:
            checking_question = current_data.get('checking_question', '')
            prompt = f"{checking_question} Please provide the answer with 'Yes' or 'No'."
            response = client.generate(prompt, image_path, **(generation_configs or {}))
            strict_accuracy = 1 if ('yes' in response.lower()) else 0

            return {
                'img_id': img_id,
                'response': response,
                'strict_accuracy': strict_accuracy
            }

        except Exception:
            # If fail, we say REJECT
            return {
                'img_id': img_id,
                'response': 'REJECT',
                'strict_accuracy': 0
            }

    elif task == 'spatial':
        soft_accuracies = []
        strict_accuracies = []
        detection_results = []  # To store the detection result(s)

        try:
            # Parse objects and relations
            all_objects = eval(current_data['objects'])      # e.g., ['cat', 'dog', 'person']
            spatials = eval(current_data['relations'])       # e.g., ['left', 'above', 'right']

            # Single detection call on the provided image_path
            detection_result = detection_model.single_detect(image_path, all_objects)
            detection_results.append(detection_result)

            # Unpack objects
            obj1, obj2, obj3 = all_objects
            spatial_relationships = spatials  # e.g., ['right', 'left', 'above']

            success_count = 0
            # We'll check each pair (obj_a, obj_b) with the expected relation
            spatial_checks = [
                (obj1, obj2, spatial_relationships[0]),
                (obj2, obj3, spatial_relationships[1]),
                (obj3, obj1, spatial_relationships[2])
            ]

            # Loop over each expected relation and see if it matches
            for obj_a, obj_b, expected_relation in spatial_checks:
                # Check if both objects were detected once
                if (detection_result['entity_info'][obj_a]['total_count'] == 1 and
                    detection_result['entity_info'][obj_b]['total_count'] == 1):

                    bbox1 = detection_result['entity_info'][obj_a]['coco_bbox'][0]
                    bbox2 = detection_result['entity_info'][obj_b]['coco_bbox'][0]

                    # Assume determine_relative_position(bbox1, bbox2) is defined
                    detected_relation = determine_relative_position(bbox1, bbox2)
                    if detected_relation == expected_relation:
                        success_count += 1
                else:
                    # If either object isn't detected exactly once, skip
                    continue

            total_relations = len(spatial_relationships)
            soft_accuracy = success_count / total_relations
            strict_accuracy = 1 if success_count == total_relations else 0

            soft_accuracies.append(soft_accuracy)
            strict_accuracies.append(strict_accuracy)

        except Exception as e:
            # If something fails (e.g., detection call breaks), we store a 'REJECT'
            detection_results = 'REJECT'
            soft_accuracies = [0]
            strict_accuracies = [0]

        # Compute (or just retrieve) average, but since it's a single call, it's the same value
        avg_soft_accuracy = sum(soft_accuracies) / len(soft_accuracies)
        avg_strict_accuracy = sum(strict_accuracies) / len(strict_accuracies)

        # Return or store in your results structure
        return {
            'img_id': img_id,
            'detection_results': detection_results,  # Could be 'REJECT' or a list with one detection result
            'soft_accuracy': avg_soft_accuracy,
            'strict_accuracy': avg_strict_accuracy
        }

    else:
        raise ValueError(f"Unknown task type: {task}")
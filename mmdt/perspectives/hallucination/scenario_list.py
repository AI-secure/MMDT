all_scenarios = {
    "text_to_image": {
        "counterfactual": ["attribute", "count", "identification", "spatial"],
        "distraction": ["attribute", "count", "identification", "spatial"],
        "natural": ["attribute", "count", "identification", "spatial"],
        "cooccurrence": ["attribute", "count", "identification", "spatial"],
        "misleading": ["attribute", "count", "identification", "spatial"],
        "ocr": ["complex", "contradictory", "distortion", "misleading"],
    },
    "image_to_text": {
        "counterfactual": ["attribute", "count", "identification", "spatial"],
        "distraction": ["attribute", "count", "identification", "spatial", "action"],
        "natural": ["attribute", "count", "identification", "spatial", "action"],
        "cooccurrence": ["attribute", "count", "identification", "spatial", "action"],
        "misleading": ["attribute", "count", "identification", "spatial", "action"],
        "ocr": ["contradictory", "cooccur", "doc", "scene"],
    },
}


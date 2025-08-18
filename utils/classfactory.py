def instantiate_object(class_name, config=None):
    try:
        # Dynamically import the class
        class_to_instantiate = globals()[class_name]
        # Instantiate the class with the provided values
        if config is None:
            return class_to_instantiate()
        return class_to_instantiate(config)
    
    except KeyError:
        raise ValueError(f"Class {class_name} not found.")
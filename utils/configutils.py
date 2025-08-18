def load_class_from_config(config, key=None, lib=None):
    if key:
        config = config[key]
    if lib is None:
        lib = globals()[config['lib']]
    
    type = config['type']
    attributes = config.get('attributes', {})

    # Check if the preprocessing class exists in the 'preprocessing' module
    if not hasattr(lib, type):
        raise ValueError(f"Preprocessing class '{type}' not found in sklearn.preprocessing.")

    # Get the class dynamically
    type_class = getattr(lib, type)
    return type_class(**attributes)

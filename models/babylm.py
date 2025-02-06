def get_modules_for_babylm(selection: str):
    module_options = {
        "all": get_modules_for_babylm_all_linear,
        "attention_only": get_modules_for_babylm_attention_only,
        "fully_connected_only": get_modules_for_babylm_fully_connected_only,
    }

    # Validate selection
    if selection not in module_options:
        raise ValueError(
            f"Invalid selection '{selection}'. Choose from {list(module_options.keys())}"
        )

    # Call the appropriate function
    return module_options[selection]()


def get_modules_for_babylm_all_linear():
    total_modules = []

    for i in range(8):
        total_modules.append(f"model.decoder.layers.{i}.self_attn.k_proj")
        total_modules.append(f"model.decoder.layers.{i}.self_attn.v_proj")
        total_modules.append(f"model.decoder.layers.{i}.self_attn.q_proj")
        total_modules.append(f"model.decoder.layers.{i}.self_attn.out_proj")

    for i in range(8):
        total_modules.append(f"model.decoder.layers.{i}.fc1")
        total_modules.append(f"model.decoder.layers.{i}.fc2")

    return total_modules


def get_modules_for_babylm_attention_only():
    total_modules = []

    for i in range(8):
        total_modules.append(f"model.decoder.layers.{i}.self_attn.k_proj")
        total_modules.append(f"model.decoder.layers.{i}.self_attn.v_proj")
        total_modules.append(f"model.decoder.layers.{i}.self_attn.q_proj")
        total_modules.append(f"model.decoder.layers.{i}.self_attn.out_proj")

    return total_modules


def get_modules_for_babylm_fully_connected_only():
    total_modules = []

    for i in range(8):
        total_modules.append(f"model.decoder.layers.{i}.fc1")
        total_modules.append(f"model.decoder.layers.{i}.fc2")

    return total_modules

from typing import Dict, Any, List, Tuple, Union

def nested_dict_to_string(d: Dict, indent=0) -> str:
    """
    Convert a nested dictionary to a nicely formatted string.
    Handles keys and values that are instances of classes with __repr__ implemented.

    Args:
        d (dict): The nested dictionary to convert.
        indent (int): The current indentation level (used for recursion).

    Returns:
        str: The formatted string representation of the nested dictionary.
    """
    result = []
    for key, value in d.items():
        key_repr = repr(key)       
        if isinstance(value, dict):
            value_repr = nested_dict_to_string(value, indent + 4)
        else:
            value_repr = repr(value)
            
        result.append(" " * indent + f"{key_repr}: {value_repr}")
        
    return "{\n" + ",\n".join(result) + "\n" + " " * (indent - 4) + "}" if indent else "{\n" + ",\n".join(result) + "\n}"
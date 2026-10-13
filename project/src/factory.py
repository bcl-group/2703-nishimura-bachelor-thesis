import importlib

def _get_class(package, class_name):
    module = importlib.import_module(package)
    if class_name not in module.__all__:
        raise ValueError(f"Class '{class_name}' not found in {package}")
    return getattr(module, class_name)


def get_agent_class(class_name: str):
    return _get_class("src.agents", class_name)


def get_field_class(class_name: str):
    return _get_class("src.fields", class_name)

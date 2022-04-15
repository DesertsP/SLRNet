import importlib

MODELS = {
}


def get_model(name, *args, **kwargs):
    if name in MODELS:
        return MODELS[name](*args, **kwargs)
    else:
        model = getattr(importlib.import_module(f'models.{name}'), 'Net')(*args, **kwargs)
        return model

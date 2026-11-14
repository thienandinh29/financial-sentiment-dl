"""Model registry: a config names a model, train/run.py builds it from here."""

MODELS = {}


def register_model(name):
    """Decorator: register factory(model_cfg, embedding_matrix) -> nn.Module under `name`."""
    def wrap(factory):
        MODELS[name] = factory
        return factory
    return wrap

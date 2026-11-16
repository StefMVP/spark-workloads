"""Select only the two reviewed native recipe leaves."""

def leaf(recipe):
    from . import spark_ds41f_1030, mia_tp4
    if recipe.get('engine') == 'tensorfold-mia-tp4':
        return mia_tp4
    if recipe.get('engine') == 'vllm':
        return spark_ds41f_1030
    raise ValueError('native recipe engine is unsupported')

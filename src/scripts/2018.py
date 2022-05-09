import os, json, pandas as pd


from prostatex.dataset import DataSet
from prostatex.scripts.optimization import OptimizationContext


config = '2018.json'
model_name = 'T2-TRA--DWI--KTRANS'

def load_config(file):
    with open(file) as configuration_file:
        configuration =  json.load(configuration_file)
        if isinstance(configuration["margins"],list):
            configuration["margins"] = {v:{} for v in configuration["margins"]}
        if isinstance(configuration["modalities"], list):
            configuration["modalities"] = {v: {} for v in configuration["modalities"]}
        return configuration


configuration = load_config(os.path.join(os.path.dirname(__file__), config))
model = configuration['test_models'][model_name]


ctx = OptimizationContext(
    base_dir=configuration["base_dir"],
    dataset=DataSet(base_dir=configuration["data_dir"]),
    base_settings=configuration["optimization_settings"]
)

ctx_test = OptimizationContext(
    base_dir=configuration["base_dir"],
    dataset=DataSet(base_dir=configuration["test_data_dir"]),
    base_settings=configuration["optimization_settings"]
)

name = "model_%s-%s" % ("model", model_name)
ctx.do_extract_features(
    name=name,
    extractor_settings=dict(
        column_label=configuration["column_label"],
        modalities=configuration["modalities"],
        margins=configuration["margins"],
        features=model["features"],
        scaling_levels=configuration["scaling_levels"],
        normalization=configuration["normalization"]
    )
)


name_test = "model_test_%s-%s" % ("model", model_name)
ctx_test.do_extract_features(
    name=name_test,
    extractor_settings=dict(
        column_label=configuration["column_label"],
        modalities=configuration["modalities"],
        margins=configuration["margins"],
        features=model["features"],
        scaling_levels=configuration["scaling_levels"],
        normalization=configuration["normalization"]
    )
)

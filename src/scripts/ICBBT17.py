import os, json, pandas as pd

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from prostatex.dataset import DataSet
from prostatex.scripts.optimization import OptimizationContext

def load_config(file):
    with open(file) as configuration_file:
        configuration =  json.load(configuration_file)
        if isinstance(configuration["margins"],list):
            configuration["margins"] = {v:{} for v in configuration["margins"]}
        if isinstance(configuration["modalities"], list):
            configuration["modalities"] = {v: {} for v in configuration["modalities"]}
        return configuration

configuration = load_config(os.path.join(os.path.dirname(__file__), "ICBBT17.json"))

ctx = OptimizationContext(
    base_dir=configuration["base_dir"],
    dataset=DataSet(base_dir=configuration["data_dir"]),
    base_settings=configuration["optimization_settings"]
)

classifiers_feature_selection = [ KNeighborsClassifier(n_neighbors=n) for n in configuration["knn_n"]  ]

classifiers_test = [
    *classifiers_feature_selection,
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
]

##TEST baseline
if "baseline_features" in configuration and len(configuration["baseline_features"]) > 0:
    ctx.do_fit(
        name="baseline",
        cls=classifiers_feature_selection,
        extractor_settings=dict(
            features=configuration["baseline_features"],
        )
    )

##TEST Meta
if "meta_features" in configuration and len(configuration["meta_features"]) > 0:
    ctx.do_fit(
        name="meta",
        cls=classifiers_feature_selection,
        extractor_settings=dict(
            features=configuration["meta_features"],
        )
    )

##TEST Each normalization method
if "test_features" in configuration:
    for norm_name, normalization in configuration["normalizations"].items():
        for modality in configuration["modalities"]:
            for feature_name, feature in configuration["test_features"].items():
                name = "%s_%s_%s_%s" % (modality, norm_name, "all-margins", feature_name)
                ##TEST feature on modality
                features = configuration["baseline_features"].copy()
                features[feature_name] = feature
                ctx.do_extract_features(name=name,
                    extractor_settings=dict(
                        modalities={modality:{}},
                        margins=configuration["margins"],
                        features=features,
                        normalization=normalization,
                        scaling_levels=configuration["scaling_levels"]
                    ))

                ctx.do_optimize_whole(name=name, cls=classifiers_feature_selection)
                #ctx.do_test_all(name=name, cls=classifiers_test)
                #ctx.do_test_best(name=name, cls=classifiers_test)

if "models_dynamic" in configuration:
    for name, dynamic_model in configuration['models_dynamic'].items():
        for i in range(configuration['models_dynamic_n']):
            test_name = "dynamic_%s-%s-%d" % ("model", name, i)
            ctx.do_merge_features(test_name, dynamic_model['filter'], dynamic_model['groupby'])
            ctx.do_test_all(test_name, classifiers_test)
            res = ctx.do_optimize_whole(test_name, classifiers_feature_selection)

if "models_merge" in configuration:
    for name, base_test_names in configuration['models_merge'].items():
        base_test_name = "merge_%s-%s" % ("model", name)

        for i in range(configuration['models_merge_n']):
            test_name = "merge_%s-%s-%d" % ("model", name, i)
            ctx.do_merge_best_features(test_name, base_test_names)
            ctx.do_test_all(test_name, cls=classifiers_test)
            res = ctx.do_optimize_whole(test_name, classifiers_feature_selection)

        results_manager = ctx.get_results_manager()

        for i in range(configuration['models_merge_n']):
            test_name = "merge_%s-%s-opt-%d" % ("model", name, i)
            best_base_test, _ = results_manager.get_best_test(base_test_name)
            ctx.do_reconfigure(best_base_test,test_name)
            ctx.do_test_all(test_name, cls=classifiers_test)
            res = ctx.do_optimize_whole(test_name,classifiers_feature_selection)

##TEST On all possible features
if "models" in configuration:
    for name, model in configuration["models"].items():
        name = "static_%s-%s" % ("model", name)
        ctx.do_extract_features(
            name=name,
            extractor_settings=dict(
                modalities=configuration["modalities"],
                margins=configuration["margins"],
                features=model["features"],
                scaling_levels=configuration["scaling_levels"]
            )
        )
        ctx.do_optimize_parts(name, classifiers_feature_selection)
        ctx.do_optimize_whole(name, classifiers_feature_selection)
        ctx.do_test_all(name=name, cls=classifiers_test)
        ctx.do_test_best(name=name, cls=classifiers_test)

import os, json, pandas as pd

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from prostatex.dataset import DataSet2
from prostatex.optimization.prostatex import prostatex_cohen
from prostatex.scripts.optimization import OptimizationContext

def load_config(file):
    with open(file) as configuration_file:
        configuration =  json.load(configuration_file)
        if isinstance(configuration["margins"],list):
            configuration["margins"] = {v:{} for v in configuration["margins"]}
        if isinstance(configuration["modalities"], list):
            configuration["modalities"] = {v: {} for v in configuration["modalities"]}
        return configuration

configuration = load_config(os.path.join(os.path.dirname(__file__), "ProstateX2.json"))

dataset = DataSet2(base_dir=configuration["data_dir"])
ctx = OptimizationContext(
    base_dir=configuration["base_dir"],
    dataset=dataset,
    base_settings=configuration["optimization_settings"]
)

rm = ctx.get_results_manager()
best_meta, best_features = rm.best_features(dataset, configuration["test_configuration"])


cols = best_features.columns.tolist()
cols.remove("ggg")

x = best_features.filter(items=cols)
y = best_features.filter(items=["ggg"]).as_matrix().ravel()

n_knn = 3
print(cross_val_score(KNeighborsClassifier(n_neighbors=n_knn), x, y, cv=2, scoring=make_scorer(prostatex_cohen)))

print(prostatex_cohen(y,cross_val_predict(KNeighborsClassifier(n_neighbors=n_knn), x, y, cv=StratifiedKFold(n_splits=2, random_state=0), method='predict_proba')))


dataset = DataSet2(base_dir=configuration["test_data_dir"])
ctx = OptimizationContext(
    base_dir=configuration["base_dir"],
    dataset=dataset,
    base_settings=configuration["optimization_settings"]
)
classifiers_feature_selection = [
    KNeighborsClassifier(n_neighbors=n) for n in configuration["knn_n"]
]

##TEST On all possible features
for name, model in configuration["test_models"].items():
    name = "model_%s-%s" % ("model", name)
    ctx.do_extract_features(
        name=name,
        extractor_settings=dict(
            column_label=configuration["column_label"],
            modalities=configuration["modalities"],
            margins=configuration["margins"],
            features=model["features"],
            scaling_levels=configuration["scaling_levels"]
        )
    )
    rm = ctx.get_results_manager()

    features_pre = rm.get_features(name)
    meta_pre = rm.get_meta(name)
    features_pre = features_pre.filter(items=best_features.columns)

    name_target = name+"_filtered"
    rm.save_features(name_target,meta_pre,features_pre)

    print([x for x in  set(best_features.columns.tolist()) if x not in set(features_pre.columns.tolist())])


    x = best_features.filter(items=features_pre.columns)
    y = best_features.filter(items=["ggg"]).as_matrix().ravel()
    print(cross_val_score(   KNeighborsClassifier(n_neighbors=n_knn), x, y, cv=2, scoring=make_scorer(prostatex_cohen)))

    features, meta = rm.get_features(name_target), rm.get_meta(name_target)

    cls = KNeighborsClassifier(n_neighbors=n_knn)

    out = meta.copy()
    out['ggg'] = cls.fit(x,y).predict(features_pre)

    out.to_csv("out.csv")
    a =2

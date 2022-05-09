from os.path import join, dirname
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from prostatex.dataset import DataSet
from sklearn import svm
# Configure logger
from prostatex.scripts.optimization import OptimizationContext
version = "t2-sag_margin=10_30_cv=10"

configuration = json.load(open(join(dirname(__file__), "model.json")))

classifiers_feature_selection = [ KNeighborsClassifier(n_neighbors=n) for n in configuration["knn_n"]  ]

cls = [
    *classifiers_feature_selection,
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
]


ctx = OptimizationContext(
    base_dir="log/Models",# Data directory
    dataset=DataSet(base_dir="data/ProstateX/train/"), # Configure dataset (provides parsed data)
    base_settings=configuration['optimization_settings']
)


ctx.do_extract_features(version, extractor_settings=configuration)
print(ctx.do_test_all(version, cls=cls))
#ctx.do_optimize_whole(version, cls=cls)
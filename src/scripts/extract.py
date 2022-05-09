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

modality = 't2-tra'

configuration = json.load(open(join(dirname(__file__), "extract.json")))


ds = DataSet(base_dir="data/ProstateX/train/")

findings = ds.findings()
ds.images()
ds.data()
a=2

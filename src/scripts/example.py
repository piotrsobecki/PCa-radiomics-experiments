from prostatex.dataset import DataSet
from prostatex.extractor.extraction import Extractor
from prostatex.scripts.helpers import setup_logging

# Configure logger
logger = setup_logging('prostatex')

# Data directory
data_dir = "data/ProstateX/train/"

# Configure dataset (provides parsed data)
dataset = DataSet(base_dir=data_dir)

# Show some data records (feature extractor bases on this) - first execution can take some time, because we try to locate proper DCM directories for series
data = dataset.data()

print(data.head())

# Setup extractor
extractor_settings = {
    "preprocessing": "no",  # Without preprocessing
    "normalization": "no",  # Without normalization
    "features": {  # Using those features (key format will correspond to column in features dataframe)
        "clinsig": "clinsig",  # Our classification category (label)
        "in_zone_%s": "in_zone",  # Lesion location one-hot-array
        "haralick_3d_skew_margin=%d_modality=%s_feature=%d": {  # Parametrized haralick features
            "func": "haralick",
            # Feature name (corresponds to method name available for FeatureExtraction via configured feature_providers)
            "settings": {  # Pass those settings to the function
                "reduction": "scipy.stats.skew",  # Reduction function - used to reduce haralick features
                "modalities": {  # Settings specific for modality
                    "t2-sag": {  # To extract data from those modalities
                        "normalization": "stat_2",  # Use normalization for t2-sag
                        "margins": {
                            2.5: {"features": [7]},
                            5: {"features": [5, 9]},
                            10: {"features": [7]},
                            20: {"features": [4]},
                            35: {"features": [12]},
                            45: {"features": [11]}
                        },
                        "features": [],
                        "shape": 2,  # Dimensionality - 2D / 3D
                    },
                    "t2-tra": {
                        "margins": {
                            2.5: {"features": [3]},
                            5: {"features": [0, 3,6,8]},
                            10: {"features": [10]},
                            20: {"features": [0,7]},
                            35: {"features": [4,9]}
                        },
                        "features": [],
                        "shape": 2,  # Dimensionality - 2D / 3D
                    },
                    "ktrans": {
                        "margins": {
                            -1: {"features": [12]},
                            2.5: {"features": [1, 9]},
                            5: {"features": [1, 11]},
                            10: {"features": [4, 7, 11, 12]},
                            35: {"features": [3]}
                        },
                        "shape": 2,  # Dimensionality - 2D / 3D
                    },
                    "dwi-adc": {
                        "margins": {
                            -1: {"features": [3]},
                            5: {"features": [1]},
                            10: {"features": [3, 5]},
                            45: {"features": [3]}
                        },
                        "shape": 3,  # Dimensionality - 2D / 3D
                    }
                }
            }
        },
        "intensity_region_3d_margin=%d_modality=%s_feature=%s": {  # Parametrized haralick features
            "func": "statistics", # Feature name (corresponds to method name available for FeatureExtraction via configured feature_providers)
            "settings": {  # Pass those settings to the function
                "modalities": {  # Settings specific for modality
                    "t2-cor": {  # To extract data from those modalities
                        "margins": {
                            -1: {"features": ["p10","p95"]},
                            2.5: {"features":["p05","p10"]},
                            5: {"features":  ["p05"]},
                            10: {"features": ["p25"]},
                            20: {"features": ["skw"]},
                            35: {"features": ["p10","p25","p90","krt"]},
                            45: {"features": ["p25","p75"]}
                        },
                        "shape": 3,  # Dimensionality - 2D / 3D
                    }
                }
            }
        }

    }
}

# setup extractor
feature_extractor = Extractor(dataset, extractor_settings)

# actually extract features
metadata, features = feature_extractor.extract()

# print metadata (Patient ID, Finding id)
print(metadata.head())

# print features
print(features.head())

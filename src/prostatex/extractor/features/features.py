from prostatex.extractor.features.common import FeatureExtractionBase
from prostatex.extractor.features.meta import MetaFeatureExtraction
from prostatex.extractor.features.texture import TextureFeatureExtraction
from prostatex.extractor.features.voxel import VoxelFeatureExtraction


class FeatureExtraction(FeatureExtractionBase):
    """A simple example class"""

    def __init__(self,dataset, image_provider, settings):
        super().__init__(image_provider,settings)
        self.dataset=dataset
        self.settings = {
            "feature_providers":[
                TextureFeatureExtraction(image_provider=image_provider,settings=settings),
                VoxelFeatureExtraction(image_provider=image_provider,settings=settings),
                MetaFeatureExtraction(image_provider=image_provider,settings=settings)
            ],
            'margins': [],
            'features': [],
            'shape': 2,
            'column_label': None,
            "zone_map": {
                'PZ': 1,
                'AS': 2,
                'TZ': 3,
                'SV': 4
            },
            "scaling_levels":255,
            **settings
        }

    def get_label_column(self):
        return self.settings['column_label']

    def get_features(self,features):
        configured_features = []
        for key, feature in features.items():
            for feature_provider in self.settings["feature_providers"]:
                func_name = self.get_function_name(feature)
                func = getattr(feature_provider, func_name, None)
                if callable(func):
                    configured_features.append(self.setup_function(feature_provider, key, feature))
        return configured_features

    @staticmethod
    def normalize(df, columns=None):
        if columns is None:
            columns = df.columns
        result = df.copy()
        for feature_name in columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            range = max_value - min_value
            if range == 0:
                result[feature_name] = 0
            else:
                result[feature_name] = (df[feature_name] - min_value) / range
        return result

    def setup_function(self, object, key, config):
        feat = super().setup_function(object, key, config)
        self.settings = {'modalities': {}, **self.settings}
        if 'modalities' in feat['settings']:
            feat['settings']['names'] = self.dataset.get_name_mapping(feat['settings']['modalities'])
        if 'clinsig' in feat['func'].__name__:
            self.settings['column_label'] = feat['settings']['key']
        return feat
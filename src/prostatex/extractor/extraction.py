import logging

from prostatex.extractor.features.common import FeatureCollector, FeatureSet
from prostatex.extractor.features.features import FeatureExtraction
from prostatex.extractor.image.provider import ImageProvider
from prostatex.utils.indirect import SetupFunctions



class Extractor(SetupFunctions):

    def __init__(self, dataset, settings):
        super().__init__(settings)
        self.logger = logging.getLogger('prostatex')
        self.dataset=dataset

    # Extraction
    def extract(self, settings=None):
        if settings is None:
            settings = {}
        settings = {**self.settings,**settings}
        feature_set = FeatureSet()
        image_provider = ImageProvider(self.dataset, settings)
        feature_extractor = FeatureExtraction(self.dataset,image_provider, settings)
        feature_configurations = feature_extractor.get_features(settings['features'])
        for name, group in self.dataset.data().groupby(['ProxID', 'fid']):
            collector = FeatureCollector()
            for feature_configuration in feature_configurations:
                try:
                    collector.addall(self.call_func(feature_configuration, group=group))
                except:
                    self.logger.exception("Could not collect feature: " + str(feature_configuration['settings']['key']))
            feature_set.add(*name,collector)
            self.dataset.load_dcm.cache_clear()
        meta,features = feature_set.get_features()
        column_names = list(features.columns.values)
        column_label = feature_extractor.get_label_column()
        if column_label in column_names:
            column_names.remove(column_label)
            column_names.append(column_label)
        #features = FeatureExtraction.normalize(features, column_names[:-1])
        return meta, features

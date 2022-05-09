from prostatex.dataset import Model, Model2, BaseModel
from prostatex.extractor.features.common import FeatureCollector, FeatureExtractionBase


class MetaFeatureExtraction(FeatureExtractionBase):
    def __init__(self, image_provider, settings):
        super().__init__(image_provider, settings)

    def age(self, group, settings):
        return FeatureCollector({settings['key']: BaseModel(group.iloc[0]).age()})

    def weight(self, group, settings):
        return FeatureCollector({settings['key']: BaseModel(group.iloc[0]).weight()})

    def size(self, group, settings):
        return FeatureCollector({settings['key']: BaseModel(group.iloc[0]).size()})


    def clinsig(self, group, settings):
        return FeatureCollector({settings['key']: int(Model(group.iloc[0]).clinsig())})

    def ggg(self, group, settings):
        return FeatureCollector({settings['key']: Model2(group.iloc[0]).ggg()})

    def zone(self, group, settings):
        zone = settings['zone_map'][BaseModel(group.iloc[0]).zone()]
        return FeatureCollector({settings['key']: zone})

    def in_zone(self, group, settings):
        coll = FeatureCollector()
        zone = BaseModel(group.iloc[0]).zone()
        for key_z, value in settings['zone_map'].items():
            coll.add(settings['key'] % key_z, int(key_z == zone))
        return coll
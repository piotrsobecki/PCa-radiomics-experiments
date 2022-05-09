import logging

import numpy
import pandas as pd

from prostatex.extractor.image.provider import region
from prostatex.utils.indirect import SetupFunctions

from prostatex.dataset import BaseModel


class FeatureCollector:
    def __init__(self, features=None):
        if features is None:
            features = {}
        self.row = {}
        for key, value in features.items():
            self.add(key, value)

    def add(self, name, value):
        self.row[name] = value
        return self

    def addall(self, features):
        if type(features) is FeatureCollector:
            self.row.update(features.row)
        else:
            self.row.update(features)
        return self


class FeatureSet:


    def __init__(self):
        self.logger = logging.getLogger('prostatex')
        self.features = dict()
        self.len=0
        self.placeholder_val = numpy.NaN
        self.meta = {'ProxID': [], 'fid': []}

    def add(self, proxid, fid, collector):
        row = collector.row
        for key in set(self.features.keys()).union(row.keys()):
            try:
                val = row[key]
            except KeyError:
                val = self.placeholder_val
            self.features.setdefault(key, [self.placeholder_val]*self.len).append(val)
        self.meta['ProxID'].append(proxid)
        self.meta['fid'].append(fid)
        self.len = self.len + 1

    def get_features(self):
        column_names = sorted(list(self.features.keys()))
        meta = pd.DataFrame.from_dict(self.meta)
        features = pd.DataFrame.from_dict(self.features)[column_names]
        return meta, features


class FeatureExtractionBase(SetupFunctions):
    settings = {}

    def __init__(self, image_provider, settings):
        super().__init__(settings=settings)
        self.image_provider = image_provider
        self.settings = {**self.settings, **settings}
        self.logger = logging.getLogger('prostatex')

    def process_first(self,group,settings,func):
        features = FeatureCollector()
        feats = func(BaseModel(group.iloc[0]),settings)
        for key,value in feats.items():
            features.add(settings['key'] % key,value)
        return features

    def process_modality(self, group, settings,func):
        features = FeatureCollector()
        for idx, row in group.iterrows():
            model = BaseModel(row)
            if model.name() in settings["names"]:
                modality = settings["names"][model.name()]
                modality_settings = {**settings, **settings["modalities"][modality]}
                features.addall(func(modality,model,modality_settings))
        return features



    def process_modality_margin(self, group, settings, func):
        features = FeatureCollector()
        for idx, row in group.iterrows():
            model = BaseModel(row)
            if model.name() in settings["names"] or not settings["names"]:
                modality = settings["names"][model.name()]
                if isinstance(settings["modalities"], dict):
                    modality_settings = {**settings, **settings["modalities"][modality]}
                else:
                    modality_settings = {**settings}
                image = self.image_provider.image(model, modality_settings)
                x, y, z = model.ijk()
                for margin, margin_settings in modality_settings["margins"].items():
                    margin_settings = {**modality_settings, **margin_settings}
                    reg = region(shape=margin_settings['shape'], tm=image,  x=x, y=y, z=z, margin=float(margin), spacing=model.spacing())
                    features.addall(func(modality, margin, reg, settings))
        return features

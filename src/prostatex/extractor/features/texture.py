import mahotas
import numpy
from prostatex.extractor.features.features import FeatureExtractionBase
from prostatex.features import TextureFeatures
from prostatex.utils.utils import getfunc


class TextureFeatureExtraction(FeatureExtractionBase):
    def __init__(self, image_provider, settings):
        super().__init__(image_provider, settings)

    # key example: intensity_region_3D_margin=%d_modality=%s_feature=%d
    def statistics(self, group, settings):
        settings = {"features": ["p05", "p10", "p25", "p75", "p90", "p95", "avg", "std", "skw", "krt"],
                    **settings}
        def statistics(modality, margin, reg, settings):
            features =  TextureFeatures.stats(reg)
            collected = {}
            for key in features:
                collected[settings['key'] % (margin, modality, key)] = features[key]
            return collected
        return self.process_modality_margin(group, settings, statistics)


    # key example: glcm_3d_margin=%d_modality=%s_feature=%d
    def glcm(self, group, settings):
        settings = {'distances': [1],
                    'angles': [0, numpy.pi / 4, numpy.pi / 2, 3 * numpy.pi / 4],
                    'levels': self.settings['scaling_levels'] + 1,
                    'symmetric': True,
                    'normed': True,
                    'greycoprops': ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation'],
                    **settings}
        def glcm(modality, margin, reg, settings):
            features =  TextureFeatures.glcm_props(reg,
                            features=settings['features'],
                            distances=settings['distances'],
                            angles=settings['angles'],
                            levels=settings['levels'],
                            symmetric=settings['symmetric'],
                            normed=settings['normed'])
            collected = {}
            for key in features:
                collected[settings['key'] % (margin, modality, key)] = features[key]
            return collected
        return self.process_modality_margin(group, settings, glcm)

    # key example: haralick_3d_mean_margin=%d_modality=%s_feature=%d
    def haralick(self, group, settings):
        settings = {'reduction': 'numpy.mean', 'features': range(13), **settings}
        def haralick(modality, margin, reg, settings):
            try:
                features = TextureFeatures.haralick(reg.astype(numpy.uint32), reduction=getfunc(settings['reduction']))
            except ValueError:
                self.logger.warning("Could not collect haralick features for modality = %s, margin = %s "%(modality,margin))
                features = numpy.zeros(13)
            collected = {}
            for key in range(len(features)):
                collected[settings['key'] % (margin, modality, key)] = features[key]
            return collected


        return self.process_modality_margin(group, settings, haralick)

    # key example: ve_cwt_margin=%d_modality=%s_feature=%d
    def ve_dtcwt(self, group, settings):
        settings = {'key':'ve_%s_%s_%s', **settings}
        def ve_cwt(modality, margin, reg, settings):
            features = TextureFeatures.ve_dtcwt(reg.astype(numpy.uint32))
            collected = {}
            for key,value in features.items():
                collected[settings['key'] % (margin, modality, key)] = value
            return collected

        return self.process_modality_margin(group, settings, ve_cwt)
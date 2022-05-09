from prostatex.dataset import Model
from prostatex.extractor.features.common import FeatureCollector, FeatureExtractionBase

class VoxelFeatureExtraction(FeatureExtractionBase):
    def __init__(self, image_provider, settings):
        super().__init__(image_provider, settings)

    def intensity(self, group, settings):
        def int(modality,model,settings):
            image_ser = self.image_provider.image(model, settings)
            x, y, z = model.ijk()
            return {settings['key']%modality:image_ser[x, y, z]}
        return self.process_modality(group,settings,int)



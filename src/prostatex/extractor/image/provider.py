from functools import lru_cache
from math import floor, ceil

from prostatex.extractor.image.normalization import ImageNormalizationProvider
from prostatex.extractor.image.postprocessing import ImagePostprocessingProvider
from prostatex.extractor.image.preprocessing import ImagePreprocessingProvider
from prostatex.utils.indirect import SetupFunctions


class ImageProvider(SetupFunctions):

    steps = ['preprocessing','normalization','postprocessing']
    def __init__(self, dataset, settings):
        super().__init__(settings)
        self.settings = {
            "preprocessing":"no",
            "preprocessing_handler":ImagePreprocessingProvider(),
            "normalization":"no",
            "normalization_handler":ImageNormalizationProvider(),
            "postprocessing":"scale_image",
            "postprocessing_handler":ImagePostprocessingProvider(),
            "scaling_levels":255,
            **settings
        }
        self.dataset=dataset

    def process(self,img,method,configuration):
        handler = self.settings[method+'_handler']
        return self.call_func(self.setup_function(handler,method, configuration), img=img)

    @lru_cache(maxsize=4)
    def _image(self, model, **configurations):
        img = model.image(self.dataset)
        for step in self.steps:
            img = self.process(img, step, configurations[step])
        return img

    def image(self, model, settings):
        if settings is None:
            settings = {}
        return self._image(model, **self.image_settings_filter(settings))

    def image_settings_filter(self, settings=None):
        if settings is None:
            settings = {}
        return {k: val for k, val in {**self.settings, **settings}.items() if k in self.steps}

def region2d(tm, x, y, z, margin, spacing):
    if margin < 0:
        return tm[:, :, z]
    shape = tm.shape
    x1 = max(0, x - floor(margin / spacing[0]))
    x2 = min(x + ceil(margin / spacing[0]), shape[0])
    y1 = max(0, y - floor(margin / spacing[1]))
    y2 = min(y + ceil(margin / spacing[1]), shape[1])
    z = max(0, min(z, shape[2]))
    return tm[x1:x2, y1:y2, z]

def region3d(tm, x, y, z, margin, spacing):
    if margin < 0:
        return tm
    shape = tm.shape
    x1 = max(0, x - floor(margin / spacing[0]))
    x2 = min(x + ceil(margin / spacing[0]), shape[0])
    y1 = max(0, y - floor(margin / spacing[1]))
    y2 = min(y + ceil(margin / spacing[1]), shape[1])
    z1 = max(0, z - floor(margin / spacing[2]))
    z2 = min(z + ceil(margin / spacing[2]), shape[2])
    return tm[x1:x2, y1:y2, z1:z2]


def region(**kwargs):
    region_func = {2: region2d, 3: region3d}[kwargs['shape']]
    del kwargs['shape']
    return region_func(**kwargs)

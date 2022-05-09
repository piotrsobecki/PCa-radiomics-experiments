import numpy


# Normalization functions
class ImageNormalizationProvider():

    def no(self, img, settings):
        return img

    def stat_1(self, img, settings):
        return (img - img.mean()) / img.std()

    def stat_2(self, img, settings):
        return img / (numpy.median(img) + 2 * img.std())

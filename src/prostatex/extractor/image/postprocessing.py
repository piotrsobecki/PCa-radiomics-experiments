
class ImagePostprocessingProvider():

    def scale_image(self, img, settings):
        i_max = img.max()
        i_min = img.min()
        img = ((img - i_min) / (i_max - i_min)) * settings['scaling_levels']
        return img

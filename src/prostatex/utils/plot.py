from distutils.util import strtobool

from matplotlib import pyplot, cm, gridspec

def get_fig(row,slide):
    clinSig = strtobool(str(row['ClinSig']))
    finding_screenshot = mpimg.imread(screenshots_dir + prostatex.getScreenshotFileName(row))
    dcm_t = slide
    dcm_shape = dcm_t.shape
    dpi = 20
    fig = pyplot.figure(figsize=(dcm_shape[0]/dpi, 2*dcm_shape[1]/dpi), dpi=dpi)
    pyplot.subplot2grid((1,2), (0,0))
    pyplot.title('DCM')
    pyplot.imshow(dcm_t, interpolation='nearest')
    pyplot.axis('off')
    pyplot.subplot2grid((1,2), (0,1))
    pyplot.title('Reference')
    pyplot.imshow(finding_screenshot, cmap='Greys_r')
    pyplot.axis('off')
    return fig

def savefig(fig,out_dir_file, fname):
    if not os.path.exists(out_dir_file):
        os.makedirs(out_dir_file)
    fig.savefig(os.path.join(out_dir_file,fname), bbox_inches='tight')  # save the figure to file
    pyplot.close(fig)  # close the figure

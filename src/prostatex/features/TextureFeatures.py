import numpy
import mahotas
import dtcwt
from dtcwt import sampling
from scipy.stats import kurtosis, skew
from skimage.feature import greycomatrix
from skimage.feature import greycoprops

def stats(reg):
    return {
        "p05": numpy.percentile(reg, 5),
        "p10": numpy.percentile(reg, 10),
        "p25": numpy.percentile(reg, 25),
        "p75": numpy.percentile(reg, 75),
        "p90": numpy.percentile(reg, 90),
        "p95": numpy.percentile(reg, 95),
        "avg": numpy.average(reg),
        "std": numpy.std(reg.ravel()),
        "skw": skew(reg.ravel()),
        "krt": kurtosis(reg.ravel())
    }


def glcm(reg, **args):
    reg = reg.astype(int)
    return greycomatrix(reg, **args)


def glcm_props(reg, features, **args):
    glcm_calc = glcm(reg, **args)
    features_out = {}
    for greycoprop in features:
        features_out[greycoprop] = greycoprops(glcm_calc, greycoprop)[0, 0]
    return features_out


def haralick(reg, reduction):
    mhf = mahotas.features.haralick(reg, compute_14th_feature=False)
    if reduction is not None:
        mhf = reduction(mhf, axis=0)
    return mhf


def variance(pyramid):
    hp = pyramid.highpasses
    num_level = len(hp)
    num_dir = hp[0].shape[2]
    out = numpy.zeros((num_level, num_dir))
    for l in range(num_level):
        for d in range(num_dir):
            out[l, d] = numpy.var(hp[l][:, :, :, d])
    return out


def entropy(pyramid):
    hp = pyramid.highpasses
    lp = pyramid.lowpass
    num_level = len(hp)
    num_dir = hp[0].shape[2]
    out = numpy.zeros((num_level, num_dir))
    for l in range(num_level):
        for d in range(num_dir):
            A = numpy.abs(hp[l][:, :, :, d])
            pA = A / A.sum()
            out[l, d] = -numpy.sum(pA * numpy.log2(A))
    return out


def combine_by_direction(matrix):
    num_level = matrix.shape[0]
    num_dir = matrix.shape[1]
    out = numpy.zeros(num_level)
    for l in range(num_level):
        out[l] = numpy.sum(matrix[l, :]) / numpy.sum(numpy.power(matrix[l, :], 2))
    return out


def pad_zero(matrix, target_shape):
    pads = []
    # print('pad_zero',matrix.shape,target_shape)
    for n in range(len(matrix.shape)):
        dim = matrix.shape[n]
        target_dim = target_shape[n]
        pads.append((int(numpy.floor((target_dim-dim)/2)), int(numpy.ceil((target_dim-dim)/2))))
    return numpy.pad(matrix, pads, mode='constant')

def ve_dtcwt(reg):
    transform = dtcwt.Transform3d()
    reg_upsample = sampling.upsample(reg)
    if numpy.any(numpy.fmod(reg_upsample.shape, 2) != 0):
        reg_upsample = pad_zero(reg_upsample,[d + d%2 for d in reg_upsample.shape])
    reg_t = transform.forward(reg_upsample, nlevels=6)
    V = variance(reg_t)
    E = entropy(reg_t)
    V = combine_by_direction(V)
    E = combine_by_direction(E)
    out = {}
    for l in range(len(V)):
        out["variance_"+str(l)] = V[l]
    for l in range(len(E)):
        out["entropy_"+str(l)] = E[l]
    return out

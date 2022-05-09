# # Show sample finding

import sys
import scipy.misc
import pandas as pd
from matplotlib import pyplot, cm, gridspec
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow, figure
from numpy import *
import os
from PIL import Image
from distutils.util import strtobool

from prostatex.extractor.image import provider

from prostatex.dataset import DataSet, Model

sys.path.append('../src')
import prostatex




def get_img(model: Model,dcm):
    x,y,z = model.ijk()
    return array(dcm[:, :,z]).T

def get_fig(model: Model,dcm):
    x,y,z = model.ijk()
    dc='g'
    if model.clinsig():
        dc = 'r'
    finding_screenshot = mpimg.imread(screenshots_dir + model.screenshot_file_name())
    dcm_t = array(dcm[:, :,z]).T
    dcm_shape = dcm.shape
    dpi =  200
    zoom = 10
    fig = pyplot.figure(figsize=(zoom*dcm_shape[0]/dpi, zoom*2*dcm_shape[1]/dpi), dpi=dpi)
    pyplot.subplot2grid((1,3), (0,0))
    pyplot.title('DCM')
    pyplot.imshow(dcm_t, cmap='Greys_r')
    pyplot.axis('off')
    pyplot.subplot2grid((1,3), (0,1))
    pyplot.title('Finding')
    pyplot.scatter([x], [y], s=[0.1 * dcm_shape[1]], c=dc)
    pyplot.imshow(dcm_t, cmap='Greys_r')
    pyplot.axis('off')
    pyplot.subplot2grid((1,3), (0,2))
    pyplot.title('Reference')
    pyplot.imshow(finding_screenshot, cmap='Greys_r')
    pyplot.axis('off')
    return fig

def get_fig_region(model: Model,dcm):
    x, y, z = model.ijk()
    dc='g'
    if model.clinsig():
        dc = 'r'
    finding_screenshot = mpimg.imread(screenshots_dir + model.screenshot_file_name())
    dcm_r = provider.region3d(dcm,x,y,z,2,2)
    dcm_t = array(dcm_r[:, :,z]).T
    x = dcm_t.shape[0]/2
    y = dcm_t.shape[1]/2
    dcm_shape = dcm.shape
    dpi = 20
    fig = pyplot.figure(figsize=(dcm_shape[0]/dpi, 2*dcm_shape[1]/dpi), dpi=dpi)
    pyplot.subplot2grid((1,3), (0,0))
    pyplot.title('DCM')
    pyplot.imshow(dcm_t, cmap='Greys_r')
    pyplot.axis('off')
    pyplot.subplot2grid((1,3), (0,1))
    pyplot.title('Finding')
    pyplot.scatter([x], [y], s=[0.1 * dcm_shape[1]], c=dc)
    pyplot.imshow(dcm_t, cmap='Greys_r')
    pyplot.axis('off')
    pyplot.subplot2grid((1,3), (0,2))
    pyplot.title('Reference')
    pyplot.imshow(finding_screenshot, cmap='Greys_r')
    pyplot.axis('off')
    return fig


def savefig(fig,out_dir_file, fname):
    if not os.path.exists(out_dir_file):
        os.makedirs(out_dir_file)
    fig.savefig(os.path.join(out_dir_file,fname), bbox_inches='tight')  # save the figure to file
    pyplot.close(fig)  # close the figure


def save_findings(ds : DataSet):
    df = ds.data()
    for index,row in df.iterrows():
        try:
            proxId = row['ProxID']
            name = row['Name']
            fid = row['fid']
            seqName = str(row['DCMSeqName'])
            if seqName.startswith('*'):
                seqName = seqName[1:]
            clinSig = row['ClinSig']
            if clinSig:
                clinSigStr  = 'Significant'
            else:
                clinSigStr = 'Not Significant'
            model = ds.get_model(row)
            x,y,z = model.ijk()
            dcm,dss,seqNames = model.dcm(ds)
            img = get_img(model,dcm)
            dcmSerNum = model.dcmSerNum()
            file = os.path.join(out_dir,seqName,name,clinSigStr,"%s_fid-%d_x-%d_y-%d.tiff"%(proxId,fid,x,y))
            parent_dir = os.path.dirname(file)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            print(file)
            image = Image.fromarray(img)
            image.save(file)
        except Exception as e:
            print("Exception")
            print(e)
            print(row)

def save_findings_reg(ds : DataSet,margin=20):
    df = ds.data()
    for index,row in df.iterrows():
        try:
            proxId = row['ProxID']
            name = row['Name']
            fid = row['fid']
            seqName = str(row['DCMSeqName'])
            if seqName.startswith('*'):
                seqName = seqName[1:]
            clinSig = row['ClinSig']
            if clinSig:
                clinSigStr  = 'Significant'
            else:
                clinSigStr = 'Not Significant'
            model = ds.get_model(row)
            x,y,z = model.ijk()
            dcm,dss,seqNames = model.dcm(ds)
            img = get_img(model,dcm)
            img = img[x-margin:x+margin,y-margin:y+margin]
            dcmSerNum = model.dcmSerNum()
            file = os.path.join(out_dir,seqName,name,clinSigStr,"%s_fid-%d_x-%d_y-%d.tiff"%(proxId,fid,x,y))
            parent_dir = os.path.dirname(file)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            print(file)
            image = Image.fromarray(img).convert('L')
            image.save(file)
        except Exception as e:
            print("Exception")
            print(e)
            print(row)


def mark_all_findings(ds : DataSet):
    df = ds.data()
    for index,row in df.iterrows():
        try:
            proxId = row['ProxID']
            name = row['Name']
            fid = row['fid']
            seqName = str(row['DCMSeqName'])
            if seqName.startswith('*'):
                seqName = seqName[1:]
            clinSig = row['ClinSig']
            if clinSig:
                clinSigStr  = 'Significant'
            else:
                clinSigStr = 'Not Significant'
            dcmSerNum = row['DCMSerNum']
            model = ds.get_model(row)
            dcm,dss,seqNames = model.dcm(ds)
            fig = get_fig(model,dcm)
            savefig(fig,os.path.join(out_dir,seqName,clinSigStr),"%s-Finding%d-%s-%d.png"%(proxId,fid,name,dcmSerNum))
            #fig = get_fig_region(model,dcm)
            #savefig(fig,os.path.join(out_dir,seqName,clinSigStr),"%s-Finding%d-%s-%d_r.png"%(proxId,fid,name,dcmSerNum))
            pyplot.close(fig)  # close the figure
        except IndexError as e:
            print("IndexError when constructing series array")
            print(e)
            print(row)

rel = ''
train_file = rel+"data/ProstateX/train/lesion-information/ProstateX.csv"
screenshots_dir = rel+"data/ProstateX/train/screenshots/"
out_dir = rel+"target/findings-train-findings-reg/"
err_out = out_dir +"err.csv"

if __name__ == '__main__':
    ds = DataSet(base_dir="data/ProstateX/train/")
    save_findings_reg(ds)
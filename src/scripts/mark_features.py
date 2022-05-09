import pandas as pd
from matplotlib import pyplot, cm, gridspec
from numpy import *
import os
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.color import label2rgb
import prostatex
rel = '../'
train_file = rel+"data/ProstateX/train/leasion-information/ProstateX-train.csv"
screenshots_dir = rel+"data/ProstateX/train/screenshots/"
out_dir = rel+"target/findings-features/"
err_out = out_dir +"err-features.csv"

def features(df):
    df_err = pd.DataFrame(columns=df.columns)
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
            dcmOffset = int(row['DCMSerOffset'])
            dcmLen = int(row['Dim'].split('x')[2])
            dcmSerNum = row['DCMSerNum']
            dir = rel+row['DCMSerDir']
            x,y,z = [int(v) for v in row['ijk'].split(' ')]


            dcm,dss,seqNames = prostatex.load_dcm(dir, dcmOffset, dcmLen)

            dcm = prostatex.region3d(dcm, x, y, int(dcm.shape[0] / 4))

            slide = np.array(dcm[:,:,z]).T

            slide = slide * 255.0 / slide.max()
            # make segmentation using edge-detection and watershed
            edges = sobel(slide)
            markers = np.zeros_like(slide)
            foreground, background = 1, 2
            markers[slide < 30.0 / 255] = background
            markers[slide > 150.0 / 255] = foreground

            ws = watershed(edges, markers)
            seg1 = ndi.label(ws == foreground)[0]
            color1 = label2rgb(seg1, image=slide, bg_label=0)

            fig = prostatex.utils.plot.get_fig(row,color1)
            prostatex.utils.plot.savefig(fig,os.path.join(out_dir,seqName,clinSigStr),"%s-Finding%d-%s-%d.png"%(proxId,fid,name,dcmSerNum))

            pyplot.close(fig)  # close the figure
        except IndexError as e:
            print("IndexError when constructing series array")
            print(e)
            print(row)
            df_err = df_err.append(row)
    df_err.to_csv(err_out)


if __name__ == '__main__':
    df = pd.read_csv(train_file,sep=',')
    features(df)
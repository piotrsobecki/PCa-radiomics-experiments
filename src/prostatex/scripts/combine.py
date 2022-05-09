import glob, pandas as pd
import logging
import re
import numpy as np
from os.path import join, basename, dirname
from optparse import OptionParser


def smart_split(val, sep, to_unpack, default_val):
    splitted = val.split(sep)
    if len(splitted) < to_unpack:
        for i in range(to_unpack - len(splitted)):
            splitted.append(default_val)
    return splitted[:to_unpack]


def combine(base_dir):
    files = glob.glob(join(base_dir, '**', '**', 'datalog.csv'))
    i = 0
    dataframes = []
    for datalog in files:
        try:
            optimization_method = basename(dirname(datalog))
            test_name = basename(dirname(dirname(datalog)))
            modality, normalization, margins, feature = smart_split(test_name, '_', 4, '')
            feature, subfeature, subsubfeature = smart_split(feature, '-', 3, '')
            modality, submodality = smart_split(modality, '-', 2, '')
            dlog = pd.DataFrame.from_csv(datalog, sep=';')
            dlog['ID'] = pd.Series(i, index=dlog.index)
            dlog['Test'] = pd.Series('%s'%test_name, index=dlog.index)
            dlog['Max Fitness'] = dlog['Max Fitness']
            dlog['Optimization'] = pd.Series('%s'%optimization_method, index=dlog.index)
            dlog['Modality'] = pd.Series('%s'%modality, index=dlog.index)
            dlog['SubModality'] = pd.Series('%s'%submodality, index=dlog.index)
            dlog['Normalization'] = pd.Series('%s'%normalization, index=dlog.index)
            dlog['Margins'] = pd.Series('%s'%margins, index=dlog.index)
            dlog['Feature'] = pd.Series('%s'%feature, index=dlog.index)
            dlog['SubFeature'] = pd.Series('%s'%subfeature, index=dlog.index)
            dlog['SubSubFeature'] = pd.Series('%s'%subsubfeature, index=dlog.index)
            max = dlog.loc[dlog['Max Fitness'].argmax(), :].to_dict()
            del dlog['Max Fitness']
            selected_margins = []
            for name, val in max.items():
                if val and "margin" in name:
                    margin_val = float(re.search('_margin=(-?\d+(\.\d+)?)', name).group(1))
                    if margin_val not in selected_margins:
                        selected_margins.append(margin_val)
            selected_margins = sorted(selected_margins)
            selected_margins = ['%.2g' % margin for margin in selected_margins]
            max['Selected Margins'] = ','.join(selected_margins)
            dataframes.append(pd.DataFrame(max, index=[i]))
            i += 1
        except:
            logging.exception('Could not parse file: ' + datalog)
    out = pd.concat(dataframes)
    cols = out.columns.tolist()
    cols_order = [
        'ID',
        'Test',
        'Max Fitness',
        'Optimization',
        'Modality',
        'SubModality',
        'Normalization',
        'Margins',
        'Feature',
        'SubFeature',
        'SubSubFeature',
        'Selected Margins'
    ]
    for col in cols_order:
        if col in cols:
            cols.remove(col)
        else:
            logging.error('Col %s not in cols' % col)
    cols_out = [*cols_order]
    cols = ['ID',*cols]
    return out[cols_out], out[cols]


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-d", "--base_dir", dest="base_dir", help="in dir", metavar="FILE")
    parser.add_option("-o", "--out_dir", dest="out_file", help="out dir", metavar="FILE")
    (options, args) = parser.parse_args()
    combined, configs = combine(options.base_dir)
    combined.to_csv(join(options.out_file,"combined.csv"), sep=";")
    configs.to_csv(join(options.out_file,"configs.csv"), sep=";")
    pass

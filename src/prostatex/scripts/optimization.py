import logging
import numpy as np
import pandas as pd
import traceback
from os import makedirs, listdir
from os.path import join, dirname, exists, normpath, basename

from matplotlib import pyplot
from opt.feature_selection.genetic import CVGeneticFeatureSelection
from opt.feature_selection.genetic import GeneticLogHelper

from prostatex.extractor.extraction import Extractor
from prostatex.optimization.parts import ProstatexPartGeneticOptimizer
from prostatex.optimization.plot import plot_genlog, plot_datalog
from prostatex.optimization.prostatex import get_optimizer, read_features_labels, split_features
from prostatex.scripts.combine import combine
from prostatex.scripts.helpers import setup_logging


class OptimizationContext:
    def __init__(self, base_dir, dataset, base_settings):
        self.dataset = dataset

        self.base = base_settings
        self.base_dir = base_dir  # next_available_dir(base_dir)
        if not exists(self.base_dir):
            makedirs(self.base_dir)
        setup_logging('prostatex', self.base_dir)
        setup_logging('optimizer', self.base_dir)
        self.logger = logging.getLogger('prostatex')

    def extract_save(self, test_name, extractor):
        df_meta, df_features = extractor.extract()
        results_manager = self.get_results_manager()
        results_manager.save_features(test_name, df_meta, df_features)

    ##Feature extraction
    def do_extract_features(self, name, extractor_settings):
        self.logger.info('do_extract_features: ' + name)
        results_manager = self.get_results_manager()
        meta, features = results_manager.get_meta_features(name)
        if not np.all([exists(features), exists(meta)]):
            self.extract_save(name, Extractor(self.dataset, extractor_settings))
        return meta, features

    def do_merge_features(self, mf_test_name, filter, group):
        o_dir = join(self.base_dir, mf_test_name)
        results_manager = self.get_results_manager()
        meta, features = results_manager.get_meta_features(mf_test_name)
        if not exists(o_dir):
            makedirs(o_dir)
        if not exists(meta) or not exists(features):
            combined, configs = self.combine()
            filtered = combined[combined[list(filter.keys())].isin(filter).all(axis=1)]
            test_names = []
            for name, group in filtered.groupby(group):
                test_names.append(group.sort("Max Fitness", ascending=False).iloc[[0]].iloc[0]['Test'])
            meta, features = results_manager.join_best_features(self.dataset, test_names)
            results_manager.save_features(mf_test_name, meta, features)
        else:
            features = results_manager.get_features(mf_test_name)
        return features

    def do_merge_best_features(self, new_test_name, test_names):
        results_manager = self.get_results_manager()
        meta, features = results_manager.get_meta_features(new_test_name)
        if not exists(meta) or not exists(features):
            meta, features = results_manager.join_best_features(self.dataset, test_names)
            return results_manager.save_features(new_test_name, meta, features)

        else:
            return meta,features

    def do_all_merge_features(self, new_test_name, test_names):
        results_manager = self.get_results_manager()
        meta, features = results_manager.get_meta_features(new_test_name)
        if not exists(meta) or not exists(features):
            meta, features = results_manager.join_features(self.dataset, test_names)
            return results_manager.save_features(new_test_name, meta, features)

        else:
            return meta,features

    def do_reconfigure(self, base_test_name, test_name):
        o_dir = join(self.base_dir, test_name)
        results_manager = self.get_results_manager()
        if not exists(o_dir):
            makedirs(o_dir)
        if not exists(results_manager.get_features_file(test_name)):
            meta, features = results_manager.best_features(self.dataset,base_test_name)
            meta, features = results_manager.save_features(test_name, meta, features)
        return test_name

    ##Optimization
    def do_optimize_whole(self, name, cls):
        results_manager = self.get_results_manager()
        self.logger.info('do_optimize_whole: ' + name)
        o_dir = join(self.base_dir, name, "whole/")
        genlog_plot = join(o_dir, name + '_genlog.png')
        datalog_plot = join(o_dir, name + '_datalog.png')
        meta, features = results_manager.get_meta_features(name)
        if not exists(o_dir):
            makedirs(o_dir)
        if not np.all([exists(o_dir), exists(genlog_plot), exists(datalog_plot)]):
            settings = {**self.base, **dict(base_dir=o_dir, data=features)}
            optimizer = get_optimizer(CVGeneticFeatureSelection, cls, settings)
            optimizer.fit()
            self.save_plots(optimizer, genlog_plot, datalog_plot)
        return self.get_results_manager().load_config(name)

    ##Optimization by parts
    def do_optimize_parts(self, name, cls):
        results_manager = self.get_results_manager()
        self.logger.info('do_optimize_parts: ' + name)
        o_dir = join(self.base_dir, name, "parts/")
        genlog_plot = join(o_dir, name + '_genlog.png')
        datalog_plot = join(o_dir, name + '_datalog.png')
        meta, features = results_manager.get_meta_features(name)
        if not exists(o_dir):
            makedirs(o_dir)
        if not np.all([exists(o_dir), exists(genlog_plot), exists(datalog_plot)]):
            settings = {**self.base, **dict(base_dir=o_dir, data=features)}
            optimizer = get_optimizer(ProstatexPartGeneticOptimizer, cls, settings)
            optimizer.fit()
            self.save_plots(optimizer, genlog_plot, datalog_plot)
        return self.get_results_manager().load_config(name)

    ##Filter by results of optimization
    def do_filter_features(self, name):
        results = self.get_results_manager()
        o_dir = join(self.base_dir, name)
        meta,features = results.best_features(self.dataset, name)
        meta_f,features_f = join(o_dir, "meta-filtered.csv"), join(o_dir, "features-filtered.csv")
        meta.to_csv(meta_f,sep=';',index=None)
        features.to_csv(features_f,sep=';',index=None)
        return meta,features

    def do_test_all(self, name, cls):
        results_manager = self.get_results_manager()
        self.logger.info('do_test_all: ' + name)
        o_dir = join(self.base_dir, name, "all/")
        conf = join(o_dir, "conf.csv")
        if not exists(o_dir):
            makedirs(o_dir)
        if not exists(conf):
            settings = {**self.base, **dict(base_dir=o_dir, data=results_manager.get_features_file(name))}
            features, labels, type = read_features_labels(settings)
            optimizer = get_optimizer(CVGeneticFeatureSelection, cls, settings)
            head = ["Max Fitness", *list(features.columns.values)]
            vals = []
            for i in range(len(cls)):
                auc = optimizer.eval_on([cls[i]], features, labels.as_matrix().ravel())
                vals.append(dict(zip(head, [auc, *[1] * len(features.columns)])))
            out = pd.DataFrame(vals)
            out.to_csv(conf, sep=self.base['sep'])
        else:
            out = pd.DataFrame.from_csv(conf, sep=self.base['sep'])
        return out

    def do_test_best(self, name, cls):
        results = self.get_results_manager()
        self.logger.info('do_test_best: ' + name)
        o_dir = join(self.base_dir, name, "best/")
        conf = join(o_dir, "conf.csv")
        features_f = join(o_dir, "filtered.csv")
        if not exists(o_dir):
            makedirs(o_dir)
        if not exists(conf):
            features = results.best_features(self.dataset,name)
            settings = {**self.base, **dict(base_dir=o_dir, data=features_f)}
            features, labels = split_features(features)
            optimizer = get_optimizer(CVGeneticFeatureSelection, cls, settings)
            head = ["Max Fitness", *list(features.columns.values)]
            vals = []
            for i in range(len(cls)):
                auc = optimizer.eval_on([cls[i]], features, labels)
                vals.append(dict(zip(head, [auc, *[1] * len(features.columns)])))
            pd.DataFrame(vals).to_csv(conf)

    @staticmethod
    def save_plots(optimizer, genlog_plot, datalog_plot):
        genlog_fig = plot_genlog(optimizer.get_genlog())
        genlog_fig.savefig(genlog_plot)
        pyplot.close(genlog_fig)
        datalog_fig = plot_datalog(optimizer.get_datalog())
        datalog_fig.savefig(datalog_plot)
        pyplot.close(datalog_fig)

    def get_results_manager(self):
        return ResultsManager(self.base_dir, self.base['sep'])

    def combine(self):
        results_manager = self.get_results_manager()
        combined_f, configs_f = self.get_results_manager().get_combined_files()
        if not exists(combined_f) or not exists(configs_f):
            combined, configs = combine(self.base_dir)
            results_manager.save_combined(combined, configs)
        else:
            combined, configs = self.get_results_manager().get_combined()
        return combined, configs

    def do_extract_best_features(self, test_name, filter, group):
        self.do_merge_features(test_name, filter, group)
        return self.get_results_manager().get_features_file(test_name)

    def do_fit(self, name, cls, extractor_settings):
        self.logger.info('Test: ' + name)
        try:
            self.do_extract_features(name, extractor_settings)
            self.do_optimize_whole(name, cls)
        except:
            self.logger.error('Test failed:\n%s\nError:\n%s' % (name, traceback.format_exc()))


class ResultsManager():

    index_columns = ['ProxID', 'fid']


    def __init__(self, base_dir, sep):
        self.base_dir = base_dir  # next_available_dir(base_dir)
        self.sep = sep
        self.logger = logging.getLogger('prostatex')

    def save_combined(self, combined, configs):
        combined_f = join(self.base_dir, "combined.csv")
        configs_f = join(self.base_dir, "configs.csv")
        combined.to_csv(combined_f, sep=self.sep)
        configs.to_csv(configs_f, sep=self.sep)
        return combined_f, configs_f

    def load_config(self, name):
        test_dir = join(self.base_dir, name)
        parts = join(test_dir, "parts", "datalog.csv")
        whole = join(test_dir, "whole", "datalog.csv")
        frames = []
        if exists(parts):
            frames.append(GeneticLogHelper(genlog=None, datalog=parts, sep=self.sep).get_datalog())
        if exists(whole):
            frames.append(GeneticLogHelper(genlog=None, datalog=whole, sep=self.sep).get_datalog())
        if len(frames) > 0:
            return pd.concat(frames).sort_values(by="Max Fitness", ascending=False)
        return None

    def best_features(self,dataset, name):
        config = self.load_config(name)
        meta, features = self.load_meta_features(name)
        meta, features =  ResultsManager.filter_features(meta, features, config.iloc[[0]])
        features = features.set_index(self.index_columns)
        features.loc[:, dataset.get_label_column()] = dataset.findings().set_index(self.index_columns, verify_integrity=True)[dataset.get_label_column()]
        return meta,features

    def get_combined_files(self):
        return join(self.base_dir, "combined.csv"), join(self.base_dir, "configs.csv")

    def get_combined(self):
        combined_f, configs_f = self.get_combined_files()
        combined = pd.DataFrame.from_csv(combined_f, sep=self.sep)
        confs = pd.DataFrame.from_csv(configs_f, sep=self.sep)
        combined = combined.fillna('*')
        return combined, confs

    def load_meta(self, meta_out_file):
        return pd.DataFrame.from_csv(meta_out_file, sep=self.sep, index_col=None)

    def load_features(self, features_out_file):
        return pd.DataFrame.from_csv(features_out_file, sep=self.sep, index_col=None)

    def get_features_file(self, test_name):
        return join(self.base_dir, test_name, "features.csv")

    def get_meta_file(self, test_name):
        return join(self.base_dir, test_name, "meta.csv")

    def get_features_filtered_file(self, test_name):
        return join(self.base_dir, test_name, "features-filtered.csv")

    def get_meta_filtered_file(self, test_name):
        return join(self.base_dir, test_name, "meta-filtered.csv")

    def get_meta(self, test_name):
        return pd.DataFrame.from_csv(self.get_meta_file(test_name), sep=self.sep)

    def get_features(self, test_name):
        return pd.DataFrame.from_csv(self.get_features_file(test_name), sep=self.sep)

    def save_features(self, test_name, meta, features):
        features_f = self.get_features_file(test_name)
        meta_f = self.get_meta_file(test_name)
        out_dir = dirname(normpath(features_f))
        if not exists(out_dir):
            makedirs(out_dir)
        meta.to_csv(meta_f, sep=self.sep)
        features.to_csv(features_f, index=False,sep=self.sep)
        return meta_f, features_f

    def get_best_test(self,base_test_name):
        configs = []
        test_names = [basename(f) for f in listdir(self.base_dir) if basename(f).startswith(base_test_name)]
        for test_name in test_names:
            config = self.load_config(test_name)
            if config is not None:
                config.loc[:, 'Test'] = test_name
                configs.append(config)
        config = pd.concat(configs)
        best = config.sort("Max Fitness", ascending=False).iloc[[0]]
        return best.iloc[0]['Test'], best

    def join_best_features(self, dataset, test_names):
        collection = []
        for base_test_name in test_names:
            test_name, best = self.get_best_test(base_test_name)
            meta, features = self.load_meta_features(test_name)
            _, features = ResultsManager.filter_features(meta, features, best)
            collection.append(features)
        return self.join_feature_collection(dataset,collection)

    def join_features(self, dataset, test_names):
        collection = []
        for base_test_name in test_names:
            test_name, best = self.get_best_test(base_test_name)
            meta, features = self.load_meta_features(test_name)
            cols = features.columns.tolist()
            cols.remove(dataset.get_label_column())
            best.loc[:, cols] = 1
            _, features = ResultsManager.filter_features(meta, features, best)
            collection.append(features)
        return self.join_feature_collection(dataset,collection)

    def join_filtered_features(self, dataset, test_names):
        collection = []
        for test_name in test_names:
            meta, features = self.load_filtered_meta_features(test_name)
            collection.append(features)
        return self.join_feature_collection(dataset,collection)

    def join_feature_collection(self,dataset,collection):
        features = collection[0]
        for i in range(1, len(collection)):
            cols = list(set(collection[i].columns.values) - set(features.columns.values)) + self.index_columns
            features = pd.merge(features, collection[i][cols], on=self.index_columns)
        features = features.set_index(self.index_columns)
        features.loc[:, dataset.get_label_column()] = dataset.findings().set_index(self.index_columns, verify_integrity=True)[dataset.get_label_column()]
        features = features.dropna()
        meta = pd.DataFrame.from_records(features.index.values.tolist(), columns=self.index_columns)
        return meta, features

    @staticmethod
    def filter_features(meta, features, configs):
        index_columns = ['ProxID', 'fid']
        collected_feats = []
        collected_metas = []
        for idx, row in configs.iterrows():
            selected = []
            for key, value in row.items():
                if value and key != 'ID' and key in features and not np.math.isnan(value):
                    selected.append(key)
            feats = features[selected]
            meta.loc[:,'ID'] = pd.Series(range(len(meta.index)), index=meta.index)
            feats.loc[:,'ID'] = pd.Series(range(len(feats.index)), index=meta.index)
            meta = meta.set_index('ID')
            feats = feats.set_index('ID')
            pd.concat([meta, feats], axis=1)
            collected_feats.append(pd.concat([meta, features[selected]], axis=1))
            collected_metas.append(meta)
        out = collected_feats[0]
        meta_out = collected_metas[0]
        for i in range(1, len(collected_feats)):
            out = out.join(collected_feats[i], how="inner", on=index_columns)
            meta_out = meta_out.join(collected_metas[i], how="inner", on=index_columns)
        #out = out.set_index(index_columns)
        #out = out.dropna()
        return meta_out, out

    def load_meta_features(self, name):
        return self.load_meta(self.get_meta_file(name)), self.load_features(self.get_features_file(name))

    def load_filtered_meta_features(self, name):
        return self.load_meta(self.get_meta_filtered_file(name)), self.load_features(self.get_features_filtered_file(name))

    def get_meta_features(self, name):
        return self.get_meta_file(name), self.get_features_file(name)

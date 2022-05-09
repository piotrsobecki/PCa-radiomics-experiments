from collections import defaultdict
from opt.feature_selection.genetic import FeatureSelectionConfiguration, CVGeneticFeatureSelection

class ProstatexPartConfiguration(FeatureSelectionConfiguration):
    def __init__(self, individual, all_columns, column_groups):
        super().__init__(individual, all_columns)
        self.column_groups = column_groups

    def as_list(self):
        selected_ids = self.column_indices()
        return [int(id in selected_ids) for id in range(len(self.all_columns))]

    def column_indices(self):
        res_len = 0
        selected_groups = defaultdict(list)
        for idx, group in self.column_groups.items():
            group_len = len(group)
            if group_len == 1:
                selected_groups[idx].append(group[0])
            else:
                for id, cname in enumerate(group):
                    if self.individual[res_len + id]:
                        selected_groups[idx].append(cname)
                res_len += len(group)
        columns_ids = []

        for column_id, column in enumerate(self.all_columns):
            valid = True
            for idx, group in enumerate(column.split('_')):
                valid &= group in selected_groups[idx]
                if not valid:
                    break
            if valid:
                columns_ids.append(column_id)

        return columns_ids


class ProstatexPartGeneticOptimizer(CVGeneticFeatureSelection):
    def __init__(self, clf, features, labels, score_func, **settings):
        self.column_groups = self.parse_column_groups(features.columns.tolist())
        super().__init__(clf, features, labels, score_func, **settings)

    def parse_column_groups(self, columns):
        groups = defaultdict(list)
        for column in columns:
            for idx, part in enumerate(column.split('_')):
                if part not in groups[idx]:
                    groups[idx].append(part)
        return groups

    def features_len(self):
        possibilities = 0
        for key, group in self.column_groups.items():
            possibilities += len(group)
        return possibilities

    def configuration(self, individual):
        return ProstatexPartConfiguration(individual, self.features.columns, self.column_groups)
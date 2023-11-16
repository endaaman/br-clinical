import os
import math
from enum import IntEnum, auto
import re
from typing import NamedTuple
from dataclasses import dataclass
from collections import OrderedDict, namedtuple
import pickle
from glob import glob
import itertools

from tqdm import tqdm
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.markers as mmark
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skmetrics
import lightgbm as lgb

from endaaman import Timer, with_wrote
from endaaman.ml import get_global_seed, BaseMLArgs, BaseMLCLI

matplotlib.use('TkAgg')


J = os.path.join
sns.set(style='white')

def sigmoid(a):
    return 1 / (1 + math.e**-a)

def odds(p):
    return p / (1 - p)

def logit(p):
    return np.log(odds(p))

def specificity_score(y_true, y_pred):
    tn, fp, __fn, __tp = skmetrics.confusion_matrix(y_true, y_pred).flatten()
    return tn / (tn + fp)

DEFAULT_REV = '20230410'
# 27, 30, 32, 33, 36, 40, 68, 71, 95
DEFAULT_SEED = 36


plain_primary_cols = [
    '非造影超音波/原発巣_BIRADS',
    '非造影超音波/原発巣_lesion',
    '非造影超音波/原発巣_mass',
    col_pp_long := '非造影超音波/原発巣_浸潤径_最大径',
    col_pp_short := '非造影超音波/原発巣_浸潤径_短径',
    '非造影超音波/原発巣_浸潤径_第3軸径',
    '非造影超音波/原発巣_乳管内進展_最大径',
    '非造影超音波/原発巣_乳管内進展_短径',
    '非造影超音波/原発巣_乳管内進展_第3軸径',
]

enhance_primary_cols = [
    '造影超音波/原発巣_lesion',
    '造影超音波/原発巣_mass',
    '造影超音波/原発巣_TIC_動脈層',
    '造影超音波/原発巣_TIC_静脈層',
    *[f'造影超音波/原発巣_TIC_A{i}' for i in range(1, 10)],
    col_ep_long := '造影超音波/原発巣_浸潤径_最大径',
    col_ep_short := '造影超音波/原発巣_浸潤径_短径',
    '造影超音波/原発巣_浸潤径_第3軸径',
    '造影超音波/原発巣_乳管内進展_最大径',
    '造影超音波/原発巣_乳管内進展_短径',
    '造影超音波/原発巣_乳管内進展_第3軸径',
]

plain_lymph_cols = [
    *[f'非造影超音波/リンパ節_term_{i}' for i in range(1, 10)],
    '非造影超音波/リンパ節_mass',
    col_pl_long:='非造影超音波/リンパ節_lymphsize_最大径',
    col_pl_short:='非造影超音波/リンパ節_lymphsize_短径',
]

enhance_lymph_cols = [
    '造影超音波/リンパ節_TIC_動脈層',
    '造影超音波/リンパ節_TIC_静脈層',
    '造影超音波/リンパ節_mass',
    col_el_long:='造影超音波/リンパ節_lymphsize_最大径',
    col_el_short:='造影超音波/リンパ節_lymphsize_短径',
    *[f'造影超音波/リンパ節_term_{i}' for i in range(1, 10)],
    *[f'造影超音波/リンパ節_B_{i}' for i in range(1, 7)],
    # col_pl_lt_el:='el short < el short',
    # col_el_ratio:='el long / el short',
    # '造影超音波/リンパ節_PI_7',
    # '造影超音波/リンパ節_PI_実数',
]

cnn_preds_cols = [
    'CNN PE prediction'
]

cnn_features_cols = [
    f'CNN feaure {i}' for i in range(10)
]

@dataclass
class Condition:
    cols: list[str]
    long_name: str
    short_name: str

COLUMN_CONDITIONS = [
    Condition(plain_primary_cols, 'plain/primary', 'pp'),
    Condition(plain_lymph_cols, 'plain/lymph', 'pl'),
    Condition(enhance_primary_cols, 'enhance/primary', 'ep'),
    Condition(enhance_lymph_cols, 'enhance/lymph', 'el'),
    Condition(cnn_preds_cols, 'enhance/cnn', 'ec'),
    # Condition(cnn_features_cols, 'enhance/features', 'ef'),
]

def gen_code_maps():
    count = len(COLUMN_CONDITIONS)
    codes = [f'{i:0{count}b}' for i in range(1, 2**count)]
    cc = []
    for code in codes:
        cols = []
        names = []
        for i, bit in enumerate(code):
            bit = int(bit)
            if bit > 0:
                c = COLUMN_CONDITIONS[i]
                cols += c.cols
                names.append(c.short_name)
        label = '+'.join(names)
        cc.append([code, (label, cols)])
    return OrderedDict(cc)

CODE_MAP = gen_code_maps()

def is_code(s):
    return re.match(r'^[01]{' + str(len(COLUMN_CONDITIONS)) + r'}$', s)

def code_to_label(code):
    if is_code(code):
        return CODE_MAP[code][0]
    return code.upper()

def codes_to_hex(codes):
    nn = []
    for c in codes:
        if is_code(c):
            nn.append(f'{int(c, 2):1X}')
        else:
            nn.append(c)
    return ''.join(nn)



# target_col = '臨床病期_N'
target_base_col = 'リンパ節/病理_metalabel'
target_col = 'target'


def load_data(rev=DEFAULT_REV, target_thres=0, split=None, cnn_preds:str=None, cnn_features:str=None):
    df = pd.read_excel(f'data/clinical_data_{rev}.xlsx', header=[0, 1, 2])
    df.columns = [
        '_'.join([
            str(s).replace('\n', '').replace(' ', '')
            for s in c if not re.match('Unnamed', str(s))
        ])
        for c in df.columns
    ]
    df = df.dropna(subset=[target_base_col])
    df[target_col] = df[target_base_col] > target_thres
    df = df.rename(columns={'研究番号': 'id'}).set_index('id')

    if split:
        print(f'Split by {split}')
        df_sp = pd.read_excel(split, index_col=0)
        assert len(df) >= len(df_sp), f'Invalid splitter len: {len(df_sp)} vs {len(df)}'
        df_m = pd.merge(df, df_sp, left_index=True, right_index=True, how='left')
        assert len(df_m) == len(df), f'different len: {len(df_m)} vs {len(df)}'
        # df = df_m.fillna({'test': False})
        df = df_m
    else:
        print('Split by random')
        df['test'] = False
        __df_train, df_test = train_test_split(df, shuffle=True, stratify=df[target_col])
        df.loc[df_test.index, 'test'] = True

    df_p = None
    if cnn_preds:
        df_p = pd.read_excel(cnn_preds)
        df_p = df_p[df_p['id'] > 0]
        df_m = df_p \
            .drop_duplicates(subset='id', keep='first') \
            .set_index('id')[['pred']] \
            .rename(columns={'pred': cnn_preds_cols[0]})
        df = pd.merge(df, df_m, left_index=True, right_index=True, how='left')
        # if cnn_features:
        #     ii = []
        #     data = []
        #     for id, row in df_p.iterrows():
        #         feaure = np.load(J(cnn_features, f'{row["name"]}.npy'))
        #         data.append(feaure)
        #         ii.append(id)
        #     df_f = pd.DataFrame(index=ii, data=data, columns=cnn_features_cols)
        #     df = df.join(df_f)
        # else:
        #     df[cnn_features_cols] = 0
    # else:
    #     df[cnn_preds_cols[0]] = 0

    # df[col_pl_lt_el] = df[col_pl_short] < df[col_el_short]
    # df[col_el_ratio] = df[col_el_long] / df[col_el_short]
    return df


def train_single_gbm(x_train, y_train, x_valid, y_valid):
    train_set = lgb.Dataset(x_train, label=y_train, categorical_feature=[])
    valid_sets = [train_set]
    if np.any(x_valid):
        valid_data = lgb.Dataset(x_valid, label=y_valid, categorical_feature=[])
        valid_sets += [valid_data]

    model = lgb.train(
        params={
            'objective': 'binary',
            'num_threads': -1,
            'max_depth': 3,
            'bagging_seed': get_global_seed(),
            'random_state': get_global_seed(),
            'boosting': 'gbdt',
            'metric': 'auc',
            # 'metric': 'binary_logloss',
            'verbosity': -1,
            'zero_as_missing': False,
        },
        train_set=train_set,
        num_boost_round=100,
        valid_sets=valid_sets,
        # early_stopping_rounds=150,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(False)
        ],
        categorical_feature=[],
    )
    return model


def train_gbm(df, num_folds=5, reduction='median'):
    df_train = df[df['test'] < 1].drop(['test'], axis=1)
    df_test = df[df['test'] > 0].drop(['test'], axis=1)
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=get_global_seed())
    folds = folds.split(np.arange(len(df_train)), y=df_train[target_col])
    folds = list(folds)
    models = []

    importances = []
    for fold in tqdm(range(num_folds), leave=False):
        # print(f'fold {fold+1}/{num_folds}')
        df_x = df_train.drop([target_col], axis=1)
        df_y =  df_train[target_col]
        vv = [
            df_x.iloc[folds[fold][0]].values, # x_train
            df_y.iloc[folds[fold][0]].values, # y_train
            df_x.iloc[folds[fold][1]].values, # x_valid
            df_y.iloc[folds[fold][1]].values, # y_valid
        ]
        vv = [v.copy() for v in vv]
        model = train_single_gbm(*vv)
        models.append(model)

        importances.append(model.feature_importance(importance_type='gain'))

    cc = list(set(df.columns) - {target_col, 'test'})
    importance = pd.DataFrame(columns=cc, data=importances)
    mean = importance.mean(axis=0)
    importance = importance.transpose()
    importance['mean'] = mean
    importance = importance.sort_values(by='mean', ascending=False)
    importance = importance[importance.columns[[-1, *range(num_folds)]]]

    preds = []
    for model in models:
        x = df_test.drop([target_col], axis=1).values
        pred = model.predict(x, num_iteration=model.best_iteration)
        preds.append(pred)

    match reduction:
        case 'mean':
            pred = np.mean(preds, axis=0)
        case 'median':
            pred = np.median(preds, axis=0)
        case _:
            raise RuntimeError(f'Invalid reduction: {reduction}')

    gt =  df_test[target_col].values
    return Result(gt, pred), importance

def auc_ci(y_true, y_score):
    y_true = y_true.astype(bool)
    AUC = skmetrics.roc_auc_score(y_true, y_score)
    N1 = sum(y_true)
    N2 = sum(~y_true)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = np.sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    return np.clip([lower, upper], 0.0, 1.0)

def calc_metrics(gt, pred):
    fpr, tpr, thresholds = skmetrics.roc_curve(gt, pred)
    auc = skmetrics.auc(fpr, tpr)
    ci = auc_ci(gt, pred)

    ii = {}
    f1_scores = [skmetrics.f1_score(gt, pred > t) for t in thresholds]
    acc_scores = [skmetrics.accuracy_score(gt, pred > t) for t in thresholds]

    ii['f1'] = np.argmax(f1_scores)
    ii['acc'] = np.argmax(acc_scores)
    ii['youden'] = np.argmax(tpr - fpr)
    ii['top-left'] = np.argmin((- tpr + 1) ** 2 + fpr ** 2)

    scores = pd.DataFrame({
        k: {
            'acc': acc_scores[i],
            'f1': f1_scores[i],
            'recall': tpr[i],
            'specificity': -fpr[i]+1,
            'thres': thresholds[i],
        } for k, i in ii.items()
    }).transpose()
    return Metrics(fpr, tpr, thresholds, auc, ci, scores)


@dataclass
class Result:
    gt: np.ndarray
    pred: np.ndarray

@dataclass
class Metrics:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    ci: np.ndarray
    scores: pd.DataFrame

    @classmethod
    def from_result(cls, r):
        return calc_metrics(r.gt, r.pred)


@dataclass
class Experiment:
    code: str
    label: str
    result: Result
    metrics: Metrics

    @classmethod
    def from_result(cls, code, label, result):
        return Experiment(
            code,
            label,
            result,
            Metrics.from_result(result)
        )


    @classmethod
    def from_file(cls, path):
        if not os.path.exists(path):
            raise RuntimeError(f'{path} does not exist.')
        with open(path, mode='rb') as f:
            r = pickle.load(f)
        m = Metrics.from_result(r)
        code = os.path.splitext(os.path.basename(path))[0]
        return Experiment(
            code=code,
            label=code_to_label(code),
            result=r,
            metrics=m,
        )

@dataclass
class GBMExperiment(Experiment):
    importance: pd.DataFrame



def _plot(ee:list[Experiment], legends:str, dest:str, show:bool):
    ee = sorted(ee, key=lambda e: -e.metrics.auc)
    legends = legends.split(':')
    if not (all(legends) and len(legends) == len(ee)):
        legends = [e.label for e in ee]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    for e, legend in zip(ee, legends):
        m = e.metrics
        value = f'{m.auc*100:.1f}% ({m.ci[0]*100:.1f}-{m.ci[1]*100:.1f}%)'
        lines = ax.plot(
            m.fpr, m.tpr,
            label=f'{legend}={value}',
        )

    if True:
        vv = (
            ('Clinical diagnosis of CEUS', 0.467, 0.926, 'tab:blue'),
            ('Clinical diagnosis of conventional US', 0.394, 0.919, 'tab:orange'),
        )
        for (label, recall, spec, color) in vv:
            x = 1-spec
            y = recall
            scatter = ax.scatter((x, ), (y, ), color=color, label=label, s=64)
            # ax.text(x+0.02, y+0.02, label)

        lines = [
            Line2D([0], [0], label=vv[0][0],
                   markerfacecolor='tab:blue', color='w', marker='o', markersize=8),
            Line2D([0], [0], label=vv[1][0],
                   markerfacecolor='tab:orange', color='w', marker='o', markersize=8),

            Line2D([0], [0], label='ML based diagnosis of CEUS',
                   color='tab:blue'),
            Line2D([0], [0], label='ML based diagnosis of conventional US',
                   color='tab:orange'),
        ]

    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    plt.legend(loc='lower right', handles=lines)

    suffix = codes_to_hex([e.code for e in ee])
    p = J(dest, f'roc_{suffix}.png')
    print(f'wrote {p}')
    plt.savefig(p)
    if show:
        plt.show()


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        seed:int = 36
        rev:str = DEFAULT_REV
        split:str = ''
        thres:int = 0
        cnn_preds:str = Field('data/cnn-preds/p.xlsx', cli=('--cnn-preds', ))
        cnn_features:str = Field('', cli=('--cnn-features', ))
        show:bool = Field(False, cli=('--show', ))
        legends:str = Field('', cli=('--legends', ))

        @property
        def dest(self):
            return f'out{self.rev}_{self.seed}_t{self.thres}'

        @property
        def src(self):
            return J(self.dest, 'results')

    def pre_common(self, a):
        self.df_all = load_data(a.rev, a.thres,  a.split, a.cnn_preds, a.cnn_features)
        os.makedirs(J(a.dest, 'results'), exist_ok=True)

    def run_dump_train_test(self, a:CommonArgs):
        df = self.df_all[['id', 'test']]
        # df['test'] = df['test'].astype('int')
        p = J(a.dest, f'train_test_{a.seed}_{a.rev}.xlsx')
        df.to_excel(p, index=False)
        print(f'wrote {p}')


    class GbmArgs(CommonArgs):
        codes_to_plot:str = Field('11110:11000', cli=('--code', ))
        reduction:str = 'median'

    def run_gbm(self, a:GbmArgs):
        codes_to_plot = sorted(a.codes_to_plot.split(':'))

        experiments = []
        for code, (label, cols) in tqdm(CODE_MAP.items()):
            df = self.df_all[cols + [target_col, 'test']]
            result, importance = train_gbm(df, reduction=a.reduction)
            with open(J(a.dest, 'results', f'{code}.pickle'), 'wb') as f:
                pickle.dump(result, f)
            metrics = Metrics.from_result(result)

            experiments.append(GBMExperiment(
                label=label,
                result=result,
                metrics=metrics,
                code=code,
                importance=importance,
            ))

        # write importance
        with pd.ExcelWriter(J(a.dest, 'importance.xlsx'), engine='xlsxwriter') as writer:
            for e in reversed(experiments):
                e.importance.to_excel(writer, sheet_name=e.label)
                num_format = writer.book.add_format({'num_format': '#,##0.00'})
                worksheet = writer.sheets[e.label]
                worksheet.set_column(0, 0, 50, None)
                worksheet.set_column(1, 6, None, num_format)

        ee_to_plot = [e for e in experiments if ('all' in codes_to_plot or e.code in codes_to_plot)]
        _plot(ee_to_plot, a.legends, a.dest, a.show)


    class PlotArgs(CommonArgs):
        codes_to_plot:str = Field('111110:110000', cli=('--code', ))

    def run_plot(self, a):
        codes_to_plot = sorted(a.codes_to_plot.split(':'))

        if 'all' in codes_to_plot:
            paths = glob(J(a.src, '*.pickle'))
        else:
            paths = [J(a.src, f'{code}.pickle') for code in codes_to_plot]

        ee = [Experiment.from_file(p) for p in paths]
        _plot(ee, a.legends, a.dest, a.show)


    def run_scores(self, a):
        paths = glob(J(a.src, '*.pickle'))
        ee = [Experiment.from_file(p) for p in paths]

        # calc scores
        scores = {}
        for e in ee:
            scores[e.label] = {
                'auc': e.metrics.auc,
                'auc_lower': e.metrics.ci[0],
                'auc_upper': e.metrics.ci[1],
                'threshold': e.metrics.scores.loc['youden', 'thres'],
                'acc': e.metrics.scores.loc['youden', 'acc'],
                'recall': e.metrics.scores.loc['youden', 'recall'],
                'specificity': e.metrics.scores.loc['youden', 'specificity'],
            }

        # write scores
        df_score = pd.DataFrame(scores).transpose()
        df_score = df_score.sort_values(by='auc', ascending=False)
        with pd.ExcelWriter(with_wrote(J(a.dest, 'scores.xlsx')), engine='xlsxwriter') as writer:
            df_score.to_excel(writer, sheet_name='scores')
            num_format = writer.book.add_format({'num_format': '#,##0.000'})
            worksheet = writer.sheets['scores']
            worksheet.set_column(0, 0, 12, None)
            worksheet.set_column(1, 30, None, num_format)

    def run_lr(self, a):
        lr_cols = {
            # '造影超音波/リンパ節_term_1': 't1',
            '造影超音波/リンパ節_term_2': 't2',
            # '造影超音波/リンパ節_term_3': 't3',
            # '造影超音波/リンパ節_term_4': 't4',
            # '造影超音波/リンパ節_term_5': 't5',
            # '造影超音波/リンパ節_term_5': 't5',
            # '造影超音波/リンパ節_term_6': 't6',
            # '造影超音波/リンパ節_term_7': 't7',
            # '造影超音波/リンパ節_term_8': 't8',
            # '造影超音波/リンパ節_term_9': 't9',
            # '造影超音波/リンパ節_B_1': 'b1',
            '造影超音波/リンパ節_B_2': 'b2',
            # '造影超音波/リンパ節_B_3': 'b3',
            # '造影超音波/リンパ節_B_4': 'b4',
            '造影超音波/リンパ節_B_5': 'b5',
            # '造影超音波/リンパ節_B_6': 'b6',
            # col_ep_long: 'ep_long',
            # col_pl_short: 'pl_short',
            # col_el_short: 'el_short',
            # col_el_long: 'el_long',
            # col_pl_long: 'pl_long',
            target_col: 'target',
            'test': 'test',
        }
        df = self.df_all[list(lr_cols.keys())].rename(columns=lr_cols)
        print(len(df))
        # df = df.dropna()
        df = df.fillna(0.5)
        print(len(df))

        df_train = df[df['test'] < 1].drop(['test'], axis=1)
        df_test = df[df['test'] > 0].drop(['test'], axis=1)

        train_x = df_train.drop(['target'], axis=1)
        train_y = df_train['target']
        test_x = df_test.drop(['target'], axis=1)
        test_y = df_test['target']

        lr = LogisticRegression(random_state=a.seed)
        lr.fit(train_x, train_y)
        pred = lr.predict_proba(test_x)[:, 1]

        result = Result(test_y, pred)
        with open(J(a.dest, 'results', 'lr.pickle'), 'wb') as f:
            pickle.dump(result, f)

        _plot([Experiment.from_result('lr', 'LR', result)], a.legends, a.dest, a.show)


    def run_lr_search(self, a):
        lr_cols = {
            '造影超音波/リンパ節_term_1': 't1',
            '造影超音波/リンパ節_term_2': 't2',
            '造影超音波/リンパ節_term_3': 't3',
            '造影超音波/リンパ節_term_4': 't4',
            '造影超音波/リンパ節_term_5': 't5',
            '造影超音波/リンパ節_term_5': 't5',
            '造影超音波/リンパ節_term_6': 't6',
            '造影超音波/リンパ節_term_7': 't7',
            '造影超音波/リンパ節_term_8': 't8',
            '造影超音波/リンパ節_term_9': 't9',
            '造影超音波/リンパ節_B_1': 'b1',
            '造影超音波/リンパ節_B_2': 'b2',
            '造影超音波/リンパ節_B_3': 'b3',
            '造影超音波/リンパ節_B_4': 'b4',
            '造影超音波/リンパ節_B_5': 'b5',
            '造影超音波/リンパ節_B_6': 'b6',
            target_col: 'target',
            'test': 'test',
        }
        data = []
        kkk = list(itertools.combinations(list(lr_cols.keys())[0:-2], 2))
        for kk in tqdm(kkk):
            kk = list(kk) + [target_col]
            df = self.df_all[kk].dropna().rename(columns=lr_cols)
            x = df.drop(['target'], axis=1)
            y = df['target']

            pred = x.sum(axis=1) > 0.0
            # lr = LogisticRegression(random_state=a.seed)
            # lr.fit(train_x, train_y)
            # pred = lr.predict_proba(test_x)[:, 1]

            result = Result(y, pred)
            m = Metrics.from_result(result)
            data.append([kk[0], kk[1], m.auc,
                         m.scores.loc['youden', 'acc'],
                         m.scores.loc['youden', 'recall'],
                         m.scores.loc['youden', 'specificity'],
                         ])

        report = pd.DataFrame(data, columns=['k0', 'k1', 'auc',
                                             'acc', 'recall', 'specificity'])
        report.to_excel(J(a.dest, 'lr_search.xlsx'))

    def run_lr_coef(self, a):
        lr_cols = {
            '造影超音波/リンパ節_term_2': 't2',
            '造影超音波/リンパ節_B_2': 'b2',
            '造影超音波/リンパ節_B_5': 'b5',
            # col_pl_long: 'pl_long',
            # col_el_long: 'el_short',
            # col_el_long: 'el_long',
            # col_pl_short: 'pl_short',
            # col_pl_long: 'pl_short',
            target_col: 'target',
        }
        df = self.df_all[list(lr_cols.keys())].dropna().rename(columns=lr_cols)
        print(df)
        X = df.drop('target', axis=1)
        Y = df['target']

        lr = LogisticRegression(random_state=a.seed)
        lr.fit(X, Y)
        pred = lr.predict_proba(X)[:, 1]
        result = Result(Y, pred)
        m:Metrics = Metrics.from_result(result)
        print(m.auc)
        # _plot([Experiment('lr', 'LR', result, m)], False, a.dest, True)
        thres = m.scores.loc['top-left', 'thres']

        C = logit(thres)

        coef = lr.coef_[0]
        intercept = lr.intercept_[0]

        min_coef = min(coef)
        coef = coef/min_coef
        T = (C-intercept)/min_coef
        pp = dict(zip(list(lr_cols.values())[:-1], coef))

        print('pp', pp)
        print('T', T)

    def calc_scores(self, X, Y, params):
        coef = params[:-1]
        T = params[-1]
        p = (np.sum((X * coef).values, axis=1) - T) > 0
        y = Y.values
        return {
            'acc': skmetrics.accuracy_score(y_true=y, y_pred=p),
            'recall': skmetrics.recall_score(y_true=y, y_pred=p),
            'spec': specificity_score(y_true=y, y_pred=p),
        }

    def run_best_coef(self, a):
        lr_cols = {
            '造影超音波/リンパ節_term_1': 't1',
            # '造影超音波/リンパ節_term_2': 't2',
            # '造影超音波/リンパ節_term_3': 't3',
            # '造影超音波/リンパ節_term_4': 't4',
            # '造影超音波/リンパ節_B_1': 'b1',
            '造影超音波/リンパ節_B_2': 'b2',
            '造影超音波/リンパ節_B_5': 'b5',
            # '造影超音波/リンパ節_B_5': 'b5',

            # col_el_short: 'el_short',
            # col_ep_long: 'ep_long',
            target_col: 'target',
        }

        df_params = pd.DataFrame([
            [ 1.0, 1.0, 1.0, 2.9 ],
            [ 1.0, 1.0, 1.0, 1.9 ],
            [ 1.0, 1.0, 1.0, 0.9 ],

            [ 1.0, 1.0, 0.0, 1.9 ],
            [ 1.0, 0.0, 1.0, 1.9 ],
            [ 0.0, 1.0, 1.0, 1.9 ],

            [ 1.0, 1.0, 0.0, 0.9 ],
            [ 1.0, 0.0, 1.0, 0.9 ],
            [ 0.0, 1.0, 1.0, 0.9 ],

            [ 1.0, 0.0, 0.0, 0.9 ],
            [ 0.0, 1.0, 0.0, 0.9 ],
            [ 0.0, 0.0, 1.0, 0.9 ],
        ], columns=[*list((lr_cols.values()))[:3], 'threshold'])

        df = self.df_all[list(lr_cols.keys())].dropna().rename(columns=lr_cols)
        X = df.drop('target', axis=1)
        Y = df['target']
        print(X)

        scoress = []
        for __i, row in df_params.iterrows():
            scoress.append(self.calc_scores(X, Y, row.values))
        df_scores = pd.DataFrame(scoress)

        df = pd.concat([df_params, df_scores], axis=1)

        with pd.ExcelWriter(J(a.dest, 'clinical_scores.xlsx'), engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='j')
            df_params.to_excel(writer, sheet_name='params')
            df_scores.to_excel(writer, sheet_name='scores')


    def run_corr(self, a):
        cols = {
            '造影超音波/リンパ節_term_1': 't1',
            '造影超音波/リンパ節_term_2': 't2',
            '造影超音波/リンパ節_term_3': 't3',
            '造影超音波/リンパ節_term_4': 't4',
            '造影超音波/リンパ節_term_5': 't5',
            '造影超音波/リンパ節_term_5': 't5',
            '造影超音波/リンパ節_term_6': 't6',
            '造影超音波/リンパ節_term_7': 't7',
            '造影超音波/リンパ節_term_8': 't8',
            '造影超音波/リンパ節_term_9': 't9',
            '造影超音波/リンパ節_B_1': 'B1',
            '造影超音波/リンパ節_B_2': 'B2',
            '造影超音波/リンパ節_B_3': 'B3',
            '造影超音波/リンパ節_B_4': 'B4',
            '造影超音波/リンパ節_B_5': 'B5',
            '造影超音波/リンパ節_B_6': 'B6',
            col_pp_short: 'short(pl pr)',
            col_pp_long: 'long(pl pr)',
            col_ep_short: 'short(en pr)',
            col_ep_long: 'long(en pr)',
            col_pl_short: 'short(pl ly)',
            col_pl_long: 'long(pl ly)',
            col_el_short: 'short(en ly)',
            col_el_long: 'long(en ly)',
            target_col: 'target',
        }

        df = self.df_all[list(cols.keys())].dropna().rename(columns=cols)

        plt.figure(figsize=(26, 14))
        ax = sns.heatmap(df.corr(), vmax=1, vmin=-1, center=0, annot=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.subplots_adjust(bottom=0.25, left=0.2)
        plt.savefig(J(a.dest, 'corr.png'))
        if a.show:
            plt.show()

    class HistArgs(CommonArgs):
        mode:str = 'enhance'

    def run_hist(self, a:HistArgs):
        if a.mode == 'enhance':
            col = '造影超音波/リンパ節_lymphsize_短径'
        elif a.mode == 'plain':
            col = '非造影超音波/リンパ節_lymphsize_短径'
        else:
            raise RuntimeError(f'Invalid mode: {a.mode}')
        df = self.df_all[col]
        x0 = df[~self.df_all[target_col]]
        x1 = df[self.df_all[target_col]]

        # sturges
        num_bins = math.ceil(math.log2(len(df) * 2))
        # num_bins = 17
        x_max = df.max()
        x_min = df.min()
        num_bins = np.linspace(x_min, x_max, num_bins)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f'Lymph node short diameter ({a.mode})')
        ax.hist([x1, x0], label=['N>0', 'N=0'], bins=num_bins, alpha=0.6, stacked=True)
        # ax.hist(x0, bins=num_bins, alpha=0.6)
        # ax.hist(x1, bins=num_bins, alpha=0.6)
        ax.legend()
        ax.set_xticks(np.arange(0, 18, 2))
        ax.set_xlim(0, 18)

        plt.savefig(with_wrote(J(a.dest, f'hist_lymph_{a.mode}.png')))
        if a.show:
            plt.show()

    def run_demographic(self, a):
        num_to_ope_proc = [ 'Bt+SN', 'Bt+Ax', 'Bp+SN', 'Bp+Ax' ]

        num_to_hormone_therapy = [ 'なし', 'TAM (5y or 10y)', 'TAM+LHRHagonist', 'AI (5y or  10y)' ]
        num_to_chemo_therapy = [ 'なし', 'EC/AC⇒PTX/DTX', 'ddEC/AC⇒P/D', 'TC, PTXonly', 'CMF', 'full regimen + Cape', ]
        num_to_nac = [ 'なし', '内分泌療法', '化学療法', '化学療法+HER', ]
        num_to_her2_therapy = [ 'なし', 'HER', 'HER+PER', 'HER+PER', 'T-DM1' ]
        num_to_radio = [ 'なし', '温存乳房照射', 'PMRT', ]
        num_to_clinical_stage = [ 'Stgae 0', 'Stage I', 'Stage 2A', 'Stage 2B', 'Stage 3A', 'Stage 3B', 'Stage 3C', ]
        num_to_tumor_type = [
            'ductal carcimnoma in situ', 'invasive ductal carcinoma',
            'invasive lobular carcinoma', 'mucinous carcinoma', 'other types',
        ]
        num_to_her2 = [
            'negative: 0', 'negative: 1', 'netagive: 2_DISH or FISH (-)',
            'positive: 2_DISH or FISH (+)', '2_DISH or FISH unknown', 'positive',
        ]

        class CT(IntEnum):
            binary = 0
            numerical = 1
            categorical = 2

        C = namedtuple('C', ['old_name', 'name', 'type', 'map'])
        C.__new__.__defaults__ = ('', 0, None)

        patient_cols = [
            C('age', 'Age', CT.binary),
            C('性別', 'Sex(female)', CT.binary),
            C('閉経有無', 'Menopause', CT.binary),
        ]
        clinical_stage_cols = [
            C('臨床病期_T', 'T stage', CT.categorical, ['0', '1', '2', '3', '4']),
            C('臨床病期_N', 'N stage', CT.categorical, ['0', '1']),
            C('臨床病期_M', 'M stage', CT.categorical, ['0', '1']),
            C('臨床病期_Stage', 'Clinical stage', CT, num_to_clinical_stage),
        ]
        therapy_cols = [
            C('周術期治療_手術内容', 'Operation procedure', CT.categorical, num_to_ope_proc),
            C('周術期治療_術前薬物療法', 'Neoadjuvant chemotherapy', CT.categorical, num_to_nac),
            C('周術期治療_ホルモン療法', 'Hormone therapy', CT.categorical, num_to_hormone_therapy),
            C('周術期治療_化学療法', 'Chemotherapy', CT.categorical, num_to_chemo_therapy),
            C('周術期治療_抗HER2療法', 'HER2 therapy', CT.categorical, num_to_her2_therapy),
            C('周術期治療_術後放射線療法', 'Postoperative radiotherapy', CT.categorical, num_to_radio),
        ]
        pathological_cols = [
            C('原発巣/病理学検査(生検検体＞手術検体にて把握可能な項目)_Tummortype', 'tumor type', CT.categorical, num_to_tumor_type),
            C('原発巣/病理学検査(生検検体＞手術検体にて把握可能な項目)_浸潤', 'invasion', CT.binary),
            C('原発巣/病理学検査(生検検体＞手術検体にて把握可能な項目)_ER', 'ER', CT.numerical),
            C('原発巣/病理学検査(生検検体＞手術検体にて把握可能な項目)_PR', 'PR', CT.numerical),
            C('原発巣/病理学検査(生検検体＞手術検体にて把握可能な項目)_HER2', 'HER2', CT.categorical, num_to_her2),
            C('原発巣/病理学検査(生検検体＞手術検体にて把握可能な項目)_Ki-67', 'Ki-67', CT.numerical),
            C('原発巣/病理学検査(切除検体にてのみ把握可能な項目)_NG', 'Nuclear grade', CT.categorical, ['0', '1', '2', '3']),
            C('原発巣/病理学検査(切除検体にてのみ把握可能な項目)_HG', 'WHO grade', CT.categorical, ['0', '1', '2', '3']),
        ]

        prognosis_cols = [
            C('予後情報_再発', 'Recurrence', CT.binary),
        ]

        features_cols = [
            C(c, c, CT.numerical)
            for c in plain_primary_cols + plain_lymph_cols + enhance_primary_cols + enhance_lymph_cols
        ]

        target_cols = [
            C(target_base_col, 'meta label', CT.categorical, ['0', '1', '1mi', '2', '3']),
            C(target_col, 'meta > 0', CT.binary),
        ]

        all_cols = patient_cols + clinical_stage_cols + therapy_cols + pathological_cols + prognosis_cols + features_cols + target_cols

        df_base = self.df_all[[c.old_name for c in all_cols] + ['test']].rename(columns={c.old_name: c.name for c in all_cols})
        dfs = {
            'all': df_base.drop('test', axis=1),
            'train': df_base[~df_base['test']].drop('test', axis=1),
            'test': df_base[df_base['test']].drop('test', axis=1),
        }
        self.dfs = dfs

        cols_by_name = {c.name: c for c in all_cols}

        data = {}
        for t, df in dfs.items():
            data_by_t = {}
            for name in df.columns:
                col = cols_by_name[name]
                values = df[name].replace('-', np.nan).dropna().values
                value_with_target = df[[name, target_cols[1][1]]].replace('-', np.nan).dropna()
                x = values
                data_by_t[name] = {
                    'mean': np.mean(x),
                    'sum': np.sum(x),
                    'len': len(x),
                    'se': np.std(x, ddof=1) / np.sqrt(len(x)),
                    'corr': value_with_target.corr().values[0, 1],
                }
                if t == 'test':
                    values_on_train = dfs['train'][name].replace('-', np.nan).dropna().values
                    __t, p  = st.mannwhitneyu(values_on_train, values)
                    data_by_t[name]['p vs train'] = p

                if col.type == CT.categorical:
                    for i, label in enumerate(col.map):
                        x = values == i
                        data_by_t[f'{name} - {label}'] = {
                            'mean': np.mean(x),
                            'sum': np.sum(x),
                            'len': len(x),
                            'se': np.std(x, ddof=1) / np.sqrt(len(x)),
                        }
            data[t] = pd.DataFrame(data_by_t).transpose()

        with pd.ExcelWriter(J(a.dest, 'demographic.xlsx'), engine='xlsxwriter') as writer:
            for t, df in data.items():
                df.to_excel(writer, sheet_name=t)
            for t, df in dfs.items():
                df.to_excel(writer, sheet_name=f'{t}(data)')

    def run_export_split(self, a:CommonArgs):
        self.df_all[['test']].to_excel(with_wrote(f'data/train_test_split_{a.rev}_{a.seed}.xlsx'))

    def run_i(self, a):
        for c in enhance_lymph_cols:
            print(c)

cli = CLI()
if __name__ == '__main__':
    cli.run()

import os
import math
import re
from typing import NamedTuple
from dataclasses import dataclass
from collections import OrderedDict
import pickle
from glob import glob

import click
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skmetrics
import lightgbm as lgb

from endaaman import Timer, with_wrote
from endaaman.torch import fix_random_states, get_global_seed



sns.set(style='whitegrid')

def sigmoid(a):
    return 1 / (1 + math.e**-a)

def odds(p):
    return p / (1 - p)

def logit(p):
    return np.log(odds(p))

plain_primary_cols = [
    '非造影超音波/原発巣_BIRADS',
    '非造影超音波/原発巣_lesion(0,1)',
    '非造影超音波/原発巣_mass(0,1)',
    '非造影超音波/原発巣_浸潤径(mm)_最大径(長径)',
    '非造影超音波/原発巣_浸潤径(mm)_短径',
    '非造影超音波/原発巣_浸潤径(mm)_第3軸径',
    '非造影超音波/原発巣_乳管内進展(mm)_最大径(長径)',
    '非造影超音波/原発巣_乳管内進展(mm)_短径',
    '非造影超音波/原発巣_乳管内進展(mm)_第3軸径',
]

plain_lymph_cols = [
    *[f'非造影超音波/リンパ節_term_{i}' for i in range(1, 9)],
    '非造影超音波/リンパ節_mass(0,1)',
    '非造影超音波/リンパ節_lymphsize_最大径(長径)',
    '非造影超音波/リンパ節_lymphsize_短径',
]

enhance_primary_cols = [
    '造影超音波/原発巣_lesion(0,1)',
    '造影超音波/原発巣_mass(0,1)',
    '造影超音波/原発巣_TIC_動脈層',
    '造影超音波/原発巣_TIC_静脈層',
    *[f'造影超音波/原発巣_iflesion=1_A{i}' for i in range(1, 9)],
    '造影超音波/原発巣_浸潤径(mm)_最大径(長径)',
    '造影超音波/原発巣_浸潤径(mm)_短径',
    '造影超音波/原発巣_浸潤径(mm)_第3軸径',
    '造影超音波/原発巣_乳管内進展(mm)_最大径(長径)',
    '造影超音波/原発巣_乳管内進展(mm)_短径',
    '造影超音波/原発巣_乳管内進展(mm)_第3軸径',
]

enhance_lymph_cols = [
    '造影超音波/リンパ節_TIC_動脈層',
    '造影超音波/リンパ節_TIC_静脈層',
    '造影超音波/リンパ節_mass(0,1)',
    '造影超音波/リンパ節_lymphsize_最大径(長径)',
    '造影超音波/リンパ節_lymphsize_短径',
    *[f'造影超音波/リンパ節_term_{i}' for i in range(1, 9)],
    *[f'造影超音波/リンパ節_B_{i}' for i in range(1, 6)],
    # '造影超音波/リンパ節_PI_7',
    # '造影超音波/リンパ節_PI_実数',
]

COLs = [
    plain_primary_cols,
    plain_lymph_cols,
    enhance_primary_cols,
    enhance_lymph_cols,
]

NAMEs = [
    ['plain/primary', 'pp'],
    ['plain/lymph', 'pl'],
    ['enhance/primary', 'ep'],
    ['enhance/lymph', 'el'],
]

def gen_code_maps():
    codes = [f'{i:04b}' for i in range(1, 16)]
    cc = []
    for code in codes:
        cols = []
        names = []
        for i, bit in enumerate(code):
            bit = int(bit)
            if bit > 0:
                cols += COLs[i]
                names.append(NAMEs[i][1])
        label = '+'.join(names)
        cc.append([code, (label, cols)])
    return OrderedDict(cc)

CODE_MAP = gen_code_maps()

def is_code(s):
    return re.match('^[01]{4}$', s)

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



target_col = '臨床病期_N'

def load_data():
    df = pd.read_excel('data/clinical_data_20230212.xlsx', header=[0, 1, 2])
    df.columns = [
        '_'.join([
            str(s).replace('\n', '').replace(' ', '')
            for s in c if not re.match('Unnamed', str(s))
        ])
        for c in df.columns
    ]
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col] > 0

    df['test'] = False
    __df_train, df_test = train_test_split(df, shuffle=True, stratify=df[target_col])
    df.loc[df_test.index, 'test'] = True
    return df

def train_model(x_train, y_train, x_valid, y_valid):
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
            'verbosity': -1,
        },
        train_set=train_set,
        num_boost_round=10000,
        valid_sets=valid_sets,
        # early_stopping_rounds=150,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False),
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
        model = train_model(*vv)
        models.append(model)

        importances.append(model.feature_importance(importance_type='gain'))

    importance = pd.DataFrame(columns=df_train.columns[:-1], data=importances)
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
            'sensitivity': -fpr[i]+1,
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

@dataclass
class Experiment:
    code: str
    label: str
    result: Result
    metrics: Metrics

@dataclass
class GBMExperiment(Experiment):
    importance: pd.DataFrame


def load_experiments(paths):
    ee = []
    for path in paths:
        if not os.path.exists(path):
            raise RuntimeError(f'{path} does not exist.')
        with open(path, mode='rb') as f:
            r = pickle.load(f)
        m = calc_metrics(r.gt, r.pred)
        code = os.path.splitext(os.path.basename(path))[0]
        ee.append(Experiment(
            code=code,
            label=code_to_label(code),
            result=r,
            metrics=m,
        ))
    return ee


option_seed = click.option(
    '--seed',
    'seed',
    type=int,
    default=42,
)

@click.group()
def cli():
    pass

@cli.command()
@option_seed
@click.option('--dest', 'dest', default='out')
@click.option('--plot', 'codes_to_plot', default='1111:1010:1100:1000')
@click.option('--show', 'show', is_flag=True)
@click.option('--reduction', 'reduction', default='median')
def gbm(seed, dest, codes_to_plot, show, reduction):
    codes_to_plot = sorted(codes_to_plot.split(':'))
    fix_random_states(seed)
    df_all = load_data()

    os.makedirs(os.path.join(dest, 'results'), exist_ok=True)

    experiments = []
    for code, (label, cols) in tqdm(CODE_MAP.items()):
        df = df_all[cols + [target_col, 'test']]
        result, importance = train_gbm(df, reduction=reduction)
        with open(os.path.join(dest, 'results', f'{code}.pickle'), 'wb') as f:
            pickle.dump(result, f)
        metrics = calc_metrics(result.gt, result.pred)

        experiments.append(GBMExperiment(
            label=label,
            result=result,
            metrics=metrics,
            code=code,
            importance=importance,
        ))

    # write importance
    with pd.ExcelWriter(os.path.join(dest, 'importance.xlsx'), engine='xlsxwriter') as writer:
        for e in reversed(experiments):
            e.importance.to_excel(writer, sheet_name=e.label)
            num_format = writer.book.add_format({'num_format': '#,##0.00'})
            worksheet = writer.sheets[e.label]
            worksheet.set_column(0, 0, 50, None)
            worksheet.set_column(1, 6, None, num_format)

    ee_to_plot = [e for e in experiments if ('all' in codes_to_plot or e.code in codes_to_plot)]
    _plot(ee_to_plot, dest, show)


def _plot(ee:list[Experiment], dest:str, show:bool):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    for e in sorted(ee, key=lambda e: -e.metrics.auc):
        m = e.metrics
        ax.plot(
            m.fpr, m.tpr,
            label=f'{e.label}={m.auc*100:.1f}% ({m.ci[0]*100:.1f}-{m.ci[1]*100:.1f}%)'
        )

    ax.set_ylabel('tpr')
    ax.set_xlabel('fpr')
    plt.legend()

    suffix = codes_to_hex([e.code for e in ee])
    p = os.path.join(dest, f'roc_{suffix}.png')
    print(f'wrote {p}')
    plt.savefig(p)
    if show:
        plt.show()


@cli.command()
@click.option('--src', 'src', default='out/results')
@click.option('--dest', 'dest', default='out')
@click.option('--code', 'codes_to_plot', default='1111:1010:1100:1000')
@click.option('--show', 'show', is_flag=True)
def plot(src, dest, codes_to_plot, show):
    codes_to_plot = sorted(codes_to_plot.split(':'))

    if 'all' in codes_to_plot:
        paths = glob(os.path.join(src, '*.pickle'))
    else:
        paths = [os.path.join(src, f'{code}.pickle') for code in codes_to_plot]

    ee = load_experiments(paths)
    _plot(ee, dest, show)


@cli.command()
@click.option('--src', 'src', default='out/results')
@click.option('--dest', 'dest', default='out')
def scores(src, dest):
    ee = load_experiments(glob(os.path.join(src, '*.pickle')))

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
            'sensitivity': e.metrics.scores.loc['youden', 'sensitivity'],
        }

    # write scores
    df_score = pd.DataFrame(scores).transpose()
    df_score = df_score.sort_values(by='auc', ascending=False)
    with pd.ExcelWriter(os.path.join(dest, 'scores.xlsx'), engine='xlsxwriter') as writer:
        df_score.to_excel(writer, sheet_name='scores')
        num_format = writer.book.add_format({'num_format': '#,##0.000'})
        worksheet = writer.sheets['scores']
        worksheet.set_column(0, 0, 12, None)
        worksheet.set_column(1, 30, None, num_format)



@cli.command()
@option_seed
@click.option('--dest', 'dest', default='out')
@click.option('--show', 'show', is_flag=True)
def lr(seed, dest, show):
    fix_random_states(seed)
    df = load_data()

    lr_cols = {
        '造影超音波/リンパ節_B_1': 'b1',
        '造影超音波/リンパ節_B_2': 'b2',
        '造影超音波/リンパ節_lymphsize_短径': 'el_short',
        '非造影超音波/リンパ節_lymphsize_短径': 'pl_short',
        target_col: 'N',
        'test': 'test',
    }

    df = df[list(lr_cols.keys())].dropna().rename(columns=lr_cols)

    df_train = df[df['test'] < 1].drop(['test'], axis=1)
    df_test = df[df['test'] > 0].drop(['test'], axis=1)

    train_x = df_train.drop(['N'], axis=1)
    train_y = df_train['N']
    test_x = df_test.drop(['N'], axis=1)
    test_y = df_test['N']

    lr = LogisticRegression(random_state=seed)
    lr.fit(train_x, train_y)

    pred = lr.predict_proba(test_x)[:, 1]

    print('coef', lr.coef_)
    print('intercept', lr.intercept_)

    print('test_x', test_x.values[0])
    print('pred', pred[0])
    print('pred(calc)', sigmoid((test_x.values[0] * lr.coef_).sum() + lr.intercept_))

    result = Result(test_y, pred)
    os.makedirs(os.path.join(dest, 'results'), exist_ok=True)
    with open(os.path.join(dest, 'results', 'lr.pickle'), 'wb') as f:
        pickle.dump(result, f)

    params = [np.concatenate([lr.coef_[0], lr.intercept_])]
    columns = list(lr_cols.values())[:4] + ['intercept']

    pd.DataFrame(
        columns=columns,
        data=params
    ).to_excel('out/lr.xlsx')

@cli.command()
@click.option('--mode', 'mode', default='enhance')
def hist(mode):
    df_all = load_data()

    if mode == 'enhance':
        col = '造影超音波/リンパ節_lymphsize_短径'
    elif mode == 'plain':
        col = '非造影超音波/リンパ節_lymphsize_短径'
    else:
        raise RuntimeError(f'Invalid mode: {mode}')
    df = df_all[col]
    x0 = df[~df_all[target_col]]
    x1 = df[df_all[target_col]]

    # sturges
    num_bins = math.ceil(math.log2(len(df) * 2))
    # num_bins = 17
    x_max = df.max()
    x_min = df.min()
    num_bins = np.linspace(x_min, x_max, num_bins)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Lymph node short diameter ({mode})')
    ax.hist([x1, x0], label=['N>0', 'N=0'], bins=num_bins, alpha=0.6, stacked=True)
    # ax.hist(x0, bins=num_bins, alpha=0.6)
    # ax.hist(x1, bins=num_bins, alpha=0.6)
    ax.legend()
    ax.set_xticks(np.arange(0, 18, 2))
    ax.set_xlim(0, 18)

    plt.savefig(with_wrote(f'out/hist_lymph_{mode}.png'))
    plt.show()


@cli.command()
def corr():
    df_all = load_data()

    cols = {
        '造影超音波/リンパ節_B_1': 'b1',
        '造影超音波/リンパ節_B_2': 'b2',
        '造影超音波/リンパ節_B_3': 'b3',
        '造影超音波/リンパ節_B_4': 'b4',
        '造影超音波/リンパ節_B_5': 'b5',
        '造影超音波/リンパ節_B_6': 'b6',
        '造影超音波/リンパ節_lymphsize_短径': 'el_short',
        '非造影超音波/リンパ節_lymphsize_短径': 'pl_short',
        target_col: 'N',
    }

    df = df_all[list(cols.keys())].dropna().rename(columns=cols)

    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(df.corr(), vmax=1, vmin=-1, center=0, annot=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
    plt.subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig('out/corr.png')
    plt.show()

if __name__ == '__main__':
    cli()

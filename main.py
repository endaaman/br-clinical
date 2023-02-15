import os
import math
import re
from typing import NamedTuple
from dataclasses import dataclass

import click
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc as calc_auc, f1_score, accuracy_score
import lightgbm as lgb

from endaaman import Timer
from endaaman.torch import fix_random_states, get_global_seed


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

    df['test'] = 0
    __df_train, df_test = train_test_split(df, shuffle=True, stratify=df[target_col])
    df.loc[df_test.index, 'test'] = 1
    return df

def train_model(x_train, y_train, x_valid, y_valid, fold):
    train_data = lgb.Dataset(x_train, label=y_train, categorical_feature=[])
    valid_sets = [train_data]
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
        train_set=train_data,
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


def train_data(df, num_folds=5, reduction='median'):
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
        model = train_model(*vv, fold)
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
    return ModelResult(gt, pred, importance)


def auc_ci(y_true, y_score):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true > 0)
    N2 = sum(y_true < 1)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = np.sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    return np.clip([lower, upper], 0.0, 1.0)


@dataclass
class ModelResult:
    gt: np.ndarray
    pred: np.ndarray
    importance: pd.DataFrame

@dataclass
class ModelMetrics:
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
    df: pd.DataFrame

    result: ModelResult
    metrics: ModelMetrics

    def train(self, **kwargs):
        self.result = train_data(self.df, **kwargs)
        fpr, tpr, thresholds = roc_curve(self.result.gt, self.result.pred)
        auc = calc_auc(fpr, tpr)
        ci = auc_ci(self.result.gt, self.result.pred)

        ii = {}
        f1_scores = [f1_score(self.result.gt, self.result.pred > t) for t in thresholds]
        acc_scores = [accuracy_score(self.result.gt, self.result.pred > t) for t in thresholds]

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
        self.metrics = ModelMetrics(fpr, tpr, thresholds, auc, ci, scores)



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
@click.option('--plot', 'plot', default='1111:1010:1100')
@click.option('--show', 'show', is_flag=True)
@click.option('--reduction', 'reduction', default='median')
def train(seed, dest, plot, show, reduction):
    plot_codes = plot.split(':')
    fix_random_states(seed)
    df = load_data()

    experiments = []

    codes = [f'{i:04b}' for i in range(1, 16)]
    for code in codes:
        cc = []
        labels = []
        for i, bit in enumerate(code):
            bit = int(bit)
            if bit > 0:
                cc += COLs[i]
                labels.append(NAMEs[i][1])
        label = '+'.join(labels)

        experiments.append(Experiment(
            code=code,
            label=label,
            df=df[cc + [target_col, 'test']],
            result=None,
            metrics=None,
        ))

    for e in tqdm(experiments):
        e.train(reduction=reduction)

    os.makedirs(dest, exist_ok=True)

    # write scores
    scores = {}
    for e in experiments:
        scores[e.label] = {
            'auc': e.metrics.auc,
            'auc_lower': e.metrics.ci[0],
            'auc_upper': e.metrics.ci[1],
            'acc': e.metrics.scores.loc['acc', 'acc'],
            'recall(youden)': e.metrics.scores.loc['youden', 'recall'],
            'sensitivity(youden)': e.metrics.scores.loc['youden', 'sensitivity'],
        }

    # write scores
    df_score = pd.DataFrame(scores).transpose()
    df_score = df_score.sort_values(by='auc', ascending=False)
    with pd.ExcelWriter(os.path.join(dest, 'scores.xlsx'), engine='xlsxwriter') as writer:
        df_score.to_excel(writer, sheet_name='scores')
        num_format = writer.book.add_format({'num_format': '#,##0.000'})
        worksheet = writer.sheets['scores']
        worksheet.set_column(0, 0, 12, None)
        worksheet.set_column(1, 16, None, num_format)

    # write importance
    with pd.ExcelWriter(os.path.join(dest, 'importance.xlsx'), engine='xlsxwriter') as writer:
        for e in reversed(experiments):
            e.result.importance.to_excel(writer, sheet_name=e.label)
            num_format = writer.book.add_format({'num_format': '#,##0.00'})
            worksheet = writer.sheets[e.label]
            worksheet.set_column(0, 0, 50, None)
            worksheet.set_column(1, 6, None, num_format)

    # plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    for e in sorted(experiments, key=lambda e: -e.metrics.auc):
        if not ('all' in plot or e.code in plot_codes):
            continue
        ax.plot(
            e.metrics.fpr, e.metrics.tpr,
            label=f'{e.label}={e.metrics.auc*100:.1f}% ({e.metrics.ci[0]*100:.1f}-{e.metrics.ci[1]*100:.1f}%)'
        )

    ax.set_ylabel('tpr')
    ax.set_xlabel('fpr')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(dest, f'roc_{"_".join(plot_codes)}.png'))
    if show:
        plt.show()


@cli.command()
@option_seed
def hist(seed):
    df = load_data()

    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')
    np.random.seed(2018)

    col = '非造影超音波/リンパ節_lymphsize_短径'
    data = df[col]
    x0 = data[~df[target_col]]
    x1 = data[df[target_col]]

    # sturges
    num_bins = math.ceil(math.log2(len(data) * 2))
    # num_bins = 17
    x_max = data.max()
    x_min = data.min()
    num_bins = np.linspace(x_min, x_max, num_bins)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist([x1, x0], bins=num_bins, alpha=0.6, stacked=True)
    # ax.hist(x0, bins=num_bins, alpha=0.6)
    # ax.hist(x1, bins=num_bins, alpha=0.6)
    ax.set_xlabel('Lymph node short diameter')
    ax.set_xticks(np.arange(0, 18, 2))
    ax.set_xlim(0, 18)

    plt.show()

if __name__ == '__main__':
    cli()

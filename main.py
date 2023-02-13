import re

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from endaaman import Timer
from endaaman.torch import fix_random_states, get_global_seed

primary_plain = '非造影超音波'


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

plain_cols = plain_primary_cols + plain_lymph_cols

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


enhance_cols = enhance_primary_cols + enhance_lymph_cols

target_col = [
    '臨床病期_N',
]

def load_data():
    df = pd.read_excel('data/clinical_data_20230212.xlsx', header=[0, 1, 2])
    df.columns = [
        '_'.join([
            str(s).replace('\n', '').replace(' ', '')
            for s in c if not re.match('Unnamed', str(s))
        ])
        for c in df.columns
    ]
    return df

def train_model(x_train, y_train, x_valid, y_valid, fold):
    gbm_params = {
        'objective': 'binary',
        'num_threads': -1,
        'max_depth': 3,
        'bagging_seed': get_global_seed(),
        'random_state': get_global_seed(),
        'boosting': 'gbdt',
        'metric': 'auc',
        'verbosity': -1,
    }

    train_data = lgb.Dataset(x_train, label=y_train, categorical_feature=[])
    valid_sets = [train_data]
    if np.any(x_valid):
        valid_data = lgb.Dataset(x_valid, label=y_valid, categorical_feature=[])
        valid_sets += [valid_data]

    model = lgb.train(
        gbm_params, # モデルのパラメータ
        train_data, # 学習データ
        num_boost_round=10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
        valid_sets=valid_sets,
        # early_stopping_rounds=150,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False),
            lgb.log_evaluation(False)
        ],
        categorical_feature=[],
    )

    tmp = pd.DataFrame()
    tmp['feature'] = cols_feature
    tmp['importance'] = model.feature_importance(importance_type='gain')
    # tmp['importance'] = model.feature_importance(importance_type='split')
    self.df_importances.append(tmp)

    return model


def train_data(df_train, df_test, num_folds=5):
    t = Timer()
    t.start()
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=get_global_seed())
    folds = folds.split(np.arange(len(df_train)), y=df_train[target_col])
    folds = list(folds)
    models = []
    for fold in range(num_folds):
        # print(f'fold {fold+1}/{num_folds}')
        df_x = df_train.drop([target_col], axis=1)
        df_y =  df_train[target_col]
        vv = [
            df_x.iloc[folds[fold][0]], # x_train
            df_y.iloc[folds[fold][0]], # y_train
            df_x.iloc[folds[fold][1]], # x_valid
            df_y.iloc[folds[fold][1]], # y_valid
        ]
        vv = [v.copy() for v in vv]
        model = train_model(*vv, fold)
        models.append(model)
    t.end()
    print(f'Training time: {t.ms()}')
    return models


@click.group()
def cli():
    fix_random_states(42)

@cli.command()
# @click.option('--mil', 'use_mil', is_flag=True)
def train():
    df = load_data()
    dfs = {
        'p': df[plain_cols],
        'pe': df[plain_cols + enhance_cols],
    }
    print(df_p)


if __name__ == '__main__':
    cli()

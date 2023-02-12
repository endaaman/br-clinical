import re

import click
import pandas as pd

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
    '造影超音波/リンパ節_PI_7',
    '造影超音波/リンパ節_PI_実数',
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


@click.group()
def cli():
    pass

@cli.command()
# @click.option('--mil', 'use_mil', is_flag=True)
def train():
    df = load_data()
    df_p = df[plain_cols]
    df_pe = df[plain_cols + enhance_cols]
    print(df_p)


if __name__ == '__main__':
    cli()

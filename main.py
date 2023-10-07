from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from freq_patterns import Transactions, Apriori, FPGrowth, AprioriPar


# noinspection PyPep8Naming
def get_and_preprocess_data() -> pd.DataFrame:
    # fetch dataset
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X: pd.DataFrame = adult.data.features
    y: pd.DataFrame = adult.data.targets
    # metadata
    # pprint(adult.metadata)

    # cleaning the data, to clean cells that are not right
    X = X.replace('?', np.nan)
    y = y.replace('.', '')

    # we hope to extract features that are categorical, not numerical
    # and also, leave y alone
    feat = adult.variables
    mask = feat['type'].isin({'Categorical', 'Binary'})
    mask2 = feat['role'].isin({'Feature'})
    feat = feat[mask & mask2]['name'].tolist()

    # extract features from X
    data = X[feat].join(y, how='outer')

    return data


# noinspection PyPep8Naming
def main(models: list[Transactions] = None, running_epochs: int = 1):
    if models is None:
        models = [Apriori, FPGrowth, AprioriPar]
    data = get_and_preprocess_data()

    # convert to transactions
    T = Transactions(iterator=data.values.tolist())
    # pprint(T.transactions[10:15])
    # print(len(T.transactions))
    
    configs = [(1e-2, 9e-1), (2e-2, 8e-1), (5e-2, 5e-1), (1e-1, 5e-1)]
    configs = [(5e-1, 9e-1)]
    print('sup,conf,'+",".join(m.__name__ for m in models))
    for sup, conf in configs:
        for _ in range(running_epochs):
            print(f'{sup},{conf}', end="")
            for model in models:
                m: Transactions = model(T.transactions)
                _, _, timer_ns = m.timer(sup, conf)
                print(f',{timer_ns}', end="")
            print()


if __name__ == '__main__':
    main()
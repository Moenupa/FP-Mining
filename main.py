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
def main(run_apriori: bool = True, run_apriori_par: bool = True, run_fpgrowth: bool = True, running_epochs: int = 10):
    data = get_and_preprocess_data()

    # convert to transactions
    T = Transactions(iterator=data.values.tolist())
    # pprint(T.transactions[10:15])
    # print(len(T.transactions))
    
    configs = [(1e-2, 9e-1), (2e-2, 8e-1), (5e-2, 5e-1), (1e-1, 5e-1)]

    for sup, conf in configs:
        for _ in range(running_epochs):
            print(f'[sup={sup}, conf={conf}] elapsed time (ns):')
            if run_apriori:
                apriori = Apriori(T.transactions)
                _, _, timer_ns = apriori.timer(sup, conf)
                print(f'Apriori:    {timer_ns}')

            if run_apriori_par:
                apriori_par = AprioriPar(T.transactions)
                _, _, timer_ns = apriori_par.timer(sup, conf)
                print(f'AprioriPar: {timer_ns}')

            if run_fpgrowth:
                fpgrowth = FPGrowth(T.transactions)
                _, _, timer_ns = fpgrowth.timer(sup, conf)
                print(f'FPGrowth:   {timer_ns}')

            # Transactions.pprint(patterns_ap, [])
            # Transactions.pprint(patterns_fp, [])
            # Transactions.pprint(patterns_ap_par, [])


if __name__ == '__main__':
    main()
from ucimlrepo import fetch_ucirepo
from pprint import pprint
import pandas as pd
import numpy as np
from utils import Transactions
from apriori import Apriori
from fp_growth import FPGrowth
from apriori_par import AprioriPar


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
def main(run_apriori: bool = False, run_apriori_par: bool = False, run_fpgrowth: bool = False):
    data = get_and_preprocess_data()

    # convert to transactions
    T = Transactions(iterator=data.values.tolist())
    # pprint(T.transactions[10:15])
    # print(len(T.transactions))

    for sup, conf in [(1e-2, 9e-1), ]:
        print(f'[sup={sup}, conf={conf}] elapsed time (ms):')
        if run_apriori:
            apriori = Apriori(T.transactions)
            patterns_ap, _, time_ap = apriori.timer(sup, conf)
            print(f'Apriori:    {time_ap // 1000}')
            del apriori, patterns_ap, time_ap

        if run_apriori_par:
            apriori_par = AprioriPar(T.transactions)
            patterns_ap_par, _, time_ap_par = apriori_par.timer(sup, conf)
            print(f'AprioriPar: {time_ap_par // 1000}')
            del apriori_par, patterns_ap_par, time_ap_par

        if run_fpgrowth:
            fpgrowth = FPGrowth(T.transactions)
            patterns_fp, _, time_fp = fpgrowth.timer(sup, conf)
            print(f'FPGrowth:   {time_fp // 1000}')
            del fpgrowth

        # Transactions.pprint(patterns_ap, [])
        # Transactions.pprint(patterns_fp, [])
        # Transactions.pprint(patterns_ap_par, [])


if __name__ == '__main__':
    main(True, True, False)
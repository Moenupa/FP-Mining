from datetime import datetime as dt
from freq_patterns import Transactions, Apriori, FPGrowth, AprioriPar
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import pickle
import random


# set seed to reproduce the same results
random.seed(0)


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
    feat: pd.DataFrame = adult.variables
    mask = feat['type'].isin({'Categorical', 'Binary'})
    mask2 = feat['role'].isin({'Feature'})
    feat = feat[mask & mask2]['name'].tolist()

    # extract features from X
    data = X[feat].join(y, how='outer')

    return data
    
    
def run_model(model: Transactions, sup: float, conf: float, transactions: list, save: bool = True, save_name: str = ""):
    m: Transactions = model(transactions=transactions)
    out = m.timer(sup, conf)
    print(f',{out[-1]}', end="")
    
    if not save:
        return
    
    with open(f'out/{dt.now().strftime("%Y%m%d_%H%M")}_{save_name}.pkl', 'wb') as handle:
        pickle.dump(out, handle)


# noinspection PyPep8Naming
def main(models: list[Transactions] = None, running_epochs: int = 10):
    if models is None:
        models = [Apriori, FPGrowth, AprioriPar]
    data = get_and_preprocess_data()

    # convert to transactions
    T = Transactions(iterator=data.values.tolist())
    # pprint(T.transactions[10:15])
    # print(len(T.transactions))

    configs = [(sup * 1e-2, 5e-1) for sup in [1, 2]]
               #[1, 2, 5, 10, 20, 50]]
    
    # uci adult dataset has about 40k datapoints
    n_dtpts = [1, 
               2e-1, 5e-1, 1e-1, 
               2e-2, 5e-2, 1e-2]

    # print csv header
    print('sup,conf,n_dtpt,'+",".join(m.__name__ for m in models))
    for sup, conf in configs:
        for size in n_dtpts:
            for iter in range(running_epochs):
                print(f'{sup},{conf},{size}', end="")
                
                batch = random.sample(T.transactions, int(size * T.n_transactions))
                for model in models:
                    # init the model with a random sample of datapoints
                    run_model(model, sup, conf, batch, 
                              save=(iter == 0 and size == 1), 
                              save_name=f"{sup}_{conf}_{model.__name__}")
                print()


if __name__ == '__main__':
    main()

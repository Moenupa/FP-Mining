from .apriori import Apriori
from .utils import FrequentPattern, AssociationRule, itemset, Transactions
from typing import List, Tuple, Iterable, Set
from math import ceil
from functools import partial
from multiprocessing import Pool


def compute_local_Lk(obj: Apriori, min_sup: float, k_thresh: int) -> Set[itemset]:
    return obj.compute_Lk(min_sup, k_thresh=k_thresh).cur()


# noinspection PyPep8Naming
class AprioriPar(Apriori):
    def __init__(self, transactions: list[itemset], n_partitions: int = 0) -> None:
        super().__init__(transactions=transactions)
        partition_size = self.n_transactions ** 0.75 if n_partitions <= 0 else self.n_transactions / n_partitions
        self.partition_size: int = ceil(partition_size)
        self.partitions = list(Apriori(partition) for partition in self.partition_data())

    def partition_data(self) -> Iterable[List[itemset]]:
        """
        partition the data into `n_partitions` partitions
        """
        for i in range(0, self.n_transactions, self.partition_size):
            yield self.transactions[i:i + self.partition_size]

    def run(self, min_sup: float, min_conf: float) \
            -> Tuple[List[FrequentPattern], List[AssociationRule]]:
        """
        run the apriori algorithm.
        Return both:
        - `Lk` `[(FrequentItemSet, support), ...]`
        - `rules` `[(ItemSetA, ItemSetB, confidence), ...]`
        """
        # compute Lk for each partition
        # and union them together
        freq_itemsets: set[itemset] = set()

        with Pool(8) as pool:
            local_freq_iter = pool.map_async(partial(compute_local_Lk, min_sup=min_sup, k_thresh=2), self.partitions)
            for local_freq in local_freq_iter.get():
                freq_itemsets |= local_freq
        '''for par in self.partitions:
            Lk = par.compute_Lk(min_sup, k_thresh=2)
            freq_items |= Lk.cur()'''

        Lk = self.compute_Lk(min_sup, c0=freq_itemsets)

        # stop when Lk-1 != empty, Lk == empty, so return k-1
        patterns = [
            FrequentPattern(s, self.support_lookup[s] / self.n_transactions)
            for s in Lk.prev()
        ]
        return patterns, self.association(Lk, min_conf)


if __name__ == '__main__':
    # data = Transactions.sample(['ACD', 'BCE', 'ABCE', 'BE'])
    data = Transactions.parse(['MONKEY', 'DONKEY', 'MAKE', 'MUCKY', 'COOKIE'])
    sol = AprioriPar(data)
    sol.run_and_pprint(0.6, 0.8)

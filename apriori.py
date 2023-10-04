from utils import Transactions, FrequentItemSet, CandidateSet, AssociationRule, FrequentPattern, itemset
from itertools import chain, combinations
from typing import List, Set, Tuple
from pprint import pprint


# noinspection PyPep8Naming
class Apriori(Transactions):
    @staticmethod
    def join(Lk: FrequentItemSet, k: int) -> Set[itemset]:
        # return Ck from Lk-1
        return set(itemset(i | j) for i in Lk[k - 1] for j in Lk[k - 1] if len(i | j) == k)

    @staticmethod
    def prune(Ck: CandidateSet, Lk: FrequentItemSet, k: int) -> Set[itemset]:
        if k <= 2:
            return Ck[k]
        for candidate in Ck[k].copy():
            # drop candidate in Ck if any subset is infrequent (i.e. not in Lk-1)
            for subset in combinations(candidate, k - 1):
                if itemset(subset) not in Lk[k - 1]:
                    Ck[k].remove(candidate)
                    break
        return Ck[k]

    def association(self, Lk: FrequentItemSet, min_conf: float) -> List[AssociationRule]:
        rules = []
        for Lj in Lk:
            for s in Lj:
                for subset in chain.from_iterable(
                        combinations(s, n) for n in range(1, len(s))
                ):
                    subset = itemset(subset)
                    sup = Transactions.support(
                        s, self.transactions, self.support_lookup)
                    conf = sup / self.support_lookup[subset]
                    if conf > min_conf:
                        rules.append(AssociationRule(subset, itemset(
                            s - subset), sup / self.n_transactions, conf))
        return rules

    def run(self, min_sup: float, min_conf: float) -> Tuple[List[FrequentPattern], List[AssociationRule]]:
        """
        run the apriori algorithm.
        Return both:
        - `Lk` `[(FrequentItemSet, support), ...]`
        - `rules` `[(ItemSetA, ItemSetB, confidence), ...]`
        """
        Ck = CandidateSet()
        Lk = FrequentItemSet()

        k = 1
        Ck[k] = set(itemset(i) for t in self.transactions for i in t)
        Lk[k] = set(s for s in Ck[k]
                    if Transactions.support(s, self.transactions, self.support_lookup)
                    >= min_sup * self.n_transactions)
        # self.pprint_step(Ck, Lk, k)

        while Lk:
            k += 1
            Ck[k] = Apriori.join(Lk, k)
            Ck[k] = Apriori.prune(Ck, Lk, k)
            Lk[k] = set(s for s in Ck[k]
                        if Transactions.support(s, self.transactions, self.support_lookup)
                        >= min_sup * self.n_transactions)
            # self.pprint_step(Ck, Lk, k)

        # stop when Lk-1 != empty, Lk == empty, so return k-1
        patterns = [
            FrequentPattern(s, self.support_lookup[s] / self.n_transactions)
            for s in Lk.prev()
        ]
        return patterns, self.association(Lk, min_conf)

    @staticmethod
    def pprint_step(Ck: CandidateSet, Lk: FrequentItemSet, k: int):
        print(f'\033[4mC_{k} ->\033[0m')
        pprint(Ck.readable(), compact=True)
        print(f'\033[4mL_{k} ->\033[0m')
        pprint(Lk.cur(), compact=True)


if __name__ == '__main__':
    # data = Transactions.sample(['ACD', 'BCE', 'ABCE', 'BE'])
    data = Transactions.parse(['MONKEY', 'DONKEY', 'MAKE', 'MUCKY', 'COOKIE'])
    sol = Apriori(data)
    sol.run_and_pprint(0.6, 0.8)

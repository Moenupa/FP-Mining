from typing import Iterable, List, Set, Tuple
from collections import defaultdict as ddict
from pprint import pprint


class itemset(frozenset):
    """
    A wrapper that only overrides frozenset's `__repr__` and `__str__` methods
    because frozenset's defaults are ugly, i.e.:
    `frozenset -> str: frozenset({1, 2, 3})` \\
    `itemset -> str: {1, 2, 3}`
    """

    def __repr__(self) -> str:
        return set(self).__repr__()

    def __str__(self) -> str:
        return set(self).__str__()


class ItemSetDP(object):
    """
    a general dp table for itemset
    stores S1, ..., Sk
    """

    def __init__(self, alias: str) -> None:
        self.k = 0
        self.alias = alias
        self.table: List[Set[itemset]] = [None, ]

    def __len__(self) -> int:
        return len(self.cur())

    def __bool__(self) -> bool:
        return bool(self.cur())

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self) -> Set[itemset]:
        if self.i >= self.k:
            raise StopIteration
        self.i += 1
        return self.table[self.i]

    def cur(self) -> Set[itemset]:
        return self.table[self.k]

    def __getitem__(self, k: int) -> Set[itemset]:
        return self.table[k]

    def __setitem__(self, k: int, v: Set[itemset]) -> None:
        if k == self.k + 1:
            self.k += 1
            self.table.append(v)
        else:
            raise ValueError(
                f'trying to change previous L_{k} < current L_{self.k}')

    @staticmethod
    def print_format(itemsets: Set[itemset]) -> List[Set]:
        return list(set(s) for s in itemsets)

    def readable(self) -> List[Set]:
        return ItemSetDP.print_format(self.cur())

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'{self.alias}_{self.k} -> {self.readable()}'


class CandidateSet(ItemSetDP):
    def __init__(self, alias: str = 'C') -> None:
        super().__init__(alias)

    def cur(self):
        return self.table[0]

    def __getitem__(self, k: int) -> Set[itemset]:
        assert k == self.k, 'out of sync'
        return self.table[0]

    def __setitem__(self, k: int, v: Set[itemset]) -> None:
        # we only need to store Ck only, and Ck+1 is generated from Ck
        # so only addition, never backwards
        assert k == self.k or k == self.k + 1, 'out of sync'
        self.k = k
        self.table[0] = v


class FrequentItemSet(ItemSetDP):
    """
    dp table implemented for Lk
    stores L1, ..., Lk
    """

    def __init__(self, alias: str = 'L') -> None:
        super().__init__(alias)
        # pad with a None to make index consistent with k
        # self.table: List[Set[itemset]] = [None, ]

    def prev(self) -> Set[itemset]:
        return self.table[self.k - 1]


class AssociationRule:
    """
    wrapper for association rule
    l => r [s, c]
    condition => result [support, confidence]
    """

    def __init__(self, l: itemset, r: itemset, s: float, c: float) -> None:
        self.l = l
        self.r = r
        self.s = s
        self.c = c

    def __repr__(self) -> str:
        return f'({self.l} => {self.r}, [{round(self.s, 3)}, {round(self.c, 3)}])'

    def __str__(self) -> str:
        return self.__repr__()


class FrequentPattern:
    """
    wrapper for association rule
    l, [s]
    pattern [support]
    """

    def __init__(self, l: itemset, s: float) -> None:
        self.l = l
        self.s = s

    def __repr__(self) -> str:
        return f'({self.l}, [{round(self.s, 3)}])'

    def __str__(self) -> str:
        return self.__repr__()


class Transactions(object):

    def __init__(self, transaction_iterator: Iterable[Iterable]) -> None:
        self.support_lookup: ddict[itemset, int] = ddict(int)
        self.transactions = list(itemset(t) for t in transaction_iterator)
        self.n_transactions = len(self.transactions)

    @staticmethod
    def parse(query: List[str]) -> Set[itemset]:
        return set(itemset(t) for t in query)

    @staticmethod
    def support(s: itemset, transactions: list[itemset] = None, dest: ddict[itemset, int] = None) -> int:
        if transactions is None and dest is None:
            raise ValueError('either transactions or dest must be provided')

        # if we do not need lookup, just calculate it
        if dest is None:
            return sum((s <= t) for t in transactions)
        if s not in dest:
            dest[s] = sum((s <= t) for t in transactions)
        return dest[s]

    def run(self, min_sup: float, min_conf: float) -> Tuple[List[FrequentPattern], List[AssociationRule]]:
        raise NotImplementedError

    def run_and_pprint(self, min_sup: float, min_conf: float) -> None:
        patterns, rules = self.run(min_sup, min_conf)
        print("\033[4mFrequent Itemsets [sup]\033[0m")
        pprint(patterns)
        print("\033[4mAssociation Rules [sup,conf]\033[0m")
        pprint(rules)


if __name__ == '__main__':
    # data = Transactions.sample(['ACD', 'BCE', 'ABCE', 'BE'])
    original = ['MONKEY', 'DONKEY', 'MAKE', 'MUCKY', 'COOKIE']
    parsed = Transactions.parse(original)
    print(f'{original} \n->\n{parsed}')

from utils import Transactions, AssociationRule, FrequentPattern, itemset
from typing import Dict, Tuple, List, Iterable
from collections import defaultdict as ddict
from itertools import combinations as comb


class FPNode:
    """
    A node in the FP tree. With count only, no itemset.
    because children are dict{itemset: FPNode}
    Frequency is obtained by (itemset, FPNode.freq) iterating through the children
    """

    def __init__(self, s: itemset | None, freq: int, parent: 'FPNode' = None) -> None:
        self.s = s
        self.freq = freq
        self.link: FPNode | None = None
        self.parent: FPNode | None = parent
        # this saves our effort to search for a child node
        # we have k, v => itemset, FPNode
        # where you can get its frequency by
        # k (itemset), v.freq (frequency)
        self.children: Dict[itemset, FPNode] = {}

    def __repr__(self) -> str:
        return f'({self.s} {self.freq}) -> [{self.children.values()}]'

    def __str__(self) -> str:
        return f'{self.freq} -> [{self.children.keys()}]'

    def readable(self) -> tuple[itemset, int, list]:
        return (
            self.s, self.freq, [v.readable() for k, v in self.children.items()]
        )


class HeaderTableCell:
    """
    to form the headertable:
    `dict{  itemset: HeaderTableCell  }`
    where HeaderTableCell contains:
    - freq: int, the frequency of the itemset
    - head: FPNode, the first node in the linked list
    - tail: FPNode, the last node in the linked list
    """

    def __init__(self, freq: int) -> None:
        self.freq = freq
        self.head: FPNode | None = None
        self.tail: FPNode | None = None

    def __repr__(self) -> str:
        return f'{self.freq}'


class FPTree:
    def __init__(self, transactions: List[itemset], min_sup: int, new_root: tuple[itemset | None, int]) -> None:
        self.min_sup = min_sup
        self.transactions: List[itemset] = transactions
        self.headertable: Dict[itemset, HeaderTableCell] = {}
        self.root = FPNode(*new_root)

        self.init_headertable(min_sup)
        self.init_tree()

    def sort_headertable(self, reverse: bool = True) -> None:
        self.headertable = dict(
            sorted(self.headertable.items(), key=lambda v: v[1].freq, reverse=reverse))

    def init_headertable(self, min_sup: int) -> None:
        support_lookup: ddict[itemset, int] = ddict(int)
        itemsets = set(itemset(i) for t in self.transactions for i in t)
        # build lookup table for support
        for s in itemsets:
            _ = Transactions.support(s, self.transactions, support_lookup)
        # filter out infrequent items
        self.headertable = {
            s: HeaderTableCell(freq)
            for s, freq in support_lookup.items()
            if freq >= min_sup
        }
        self.sort_headertable()

    def is_freq(self, s: itemset) -> bool:
        if s not in self.headertable:
            raise ValueError(f'{s} not in headertable')

        return self.headertable[s].freq >= self.min_sup

    def init_tree(self):
        """
        build the tree with a transaction
        this will also fill linked list in the headertable
        """
        for t in self.transactions:
            cur = self.root
            for s, cell in self.headertable.items():
                if s <= t:
                    # insert the item into the tree
                    if s in cur.children:
                        cur = cur.children[s]
                        cur.freq += 1
                    else:
                        new_node = FPNode(s, 1, cur)
                        cur.children[s] = new_node
                        FPTree.update_headertablecell(cell, new_node)
                        cur = new_node

    @staticmethod
    def update_headertablecell(cell: HeaderTableCell, node: FPNode) -> None:
        # if table has head, update tail only
        if cell.head:
            cell.tail.link = node
            cell.tail = node
            return

        # no head, then update head and tail
        cell.head = node
        cell.tail = node

    @staticmethod
    def single_path(node: FPNode) -> bool:
        """
        check if the node has a single path
        """
        if not node.children:
            return True
        if len(node.children) > 1:
            return False
        return FPTree.single_path(next(iter(node.children.values())))

    @staticmethod
    def trace_by_link(cell: HeaderTableCell) -> Iterable[FPNode]:
        """
        travel the linked list
        """
        cur = cell.head
        while cur and cur.s:
            yield cur
            cur = cur.link

    @staticmethod
    def trace_by_parent(node: FPNode) -> Iterable[FPNode]:
        """
        travel the tree by parent
        """
        cur = node.parent
        while cur and cur.s:
            yield cur
            cur = cur.parent

    def mine_patterns(self, min_sup: int) -> dict[itemset[itemset], int]:
        """
        mine the patterns
        """
        # if the tree has a single path, then we can just return the path
        if FPTree.single_path(self.root):
            return self.mine_path()

        # otherwise, we need to mine the tree
        return self.zip_patterns(self.mine_tree(min_sup))

    def mine_path(self) -> dict[itemset[itemset], int]:
        patterns: dict[itemset[itemset], int] = {}

        # If in a conditional tree,
        # the suffix is a pattern on its own.
        if self.root.s is None:
            suffix_value = itemset()
        else:
            suffix_value = self.root.s
            patterns[suffix_value] = self.root.freq

        itemsets = self.headertable.keys()
        for i in range(1, len(itemsets) + 1):
            for subset in comb(itemsets, i):
                s = itemset()
                for x in subset:
                    s = s | x
                pattern = itemset(s | suffix_value)
                patterns[pattern] = min(
                    self.headertable[itemset(x)].freq for x in s)

        return patterns

    def mine_tree(self, min_sup: int) -> dict[itemset[itemset], int]:
        patterns = ddict(int)

        # Get items in tree in reverse order of occurrences.
        for s, cell in reversed(list(self.headertable.items())):
            conditional_tree_input: List[itemset] = []

            # For each occurrence of the item, trace the path back to the root node.
            for suffix in self.trace_by_link(cell):
                alpha = itemset()
                for node in self.trace_by_parent(suffix):
                    alpha = alpha | node.s

                for i in range(suffix.freq):
                    conditional_tree_input.append(alpha)

            # Now we have the input for a subtree,
            # so construct it and grab the patterns.
            subtree = FPTree(conditional_tree_input, min_sup, (s, cell.freq))
            subtree_patterns = subtree.mine_patterns(min_sup)

            # Insert subtree patterns into main patterns dictionary.
            for pattern, freq in subtree_patterns.items():
                patterns[pattern] += freq

        return patterns

    def zip_patterns(self, patterns: dict[itemset[itemset], int]) -> dict[itemset[itemset], int]:
        """
        append the suffix to the patterns
        """
        suffix = self.root.s
        if suffix is None:
            return patterns
        return {itemset(k | suffix): v for k, v in patterns.items()}


class FPGrowth(Transactions):
    def __init__(self, transaction_iterator: Iterable[Iterable]) -> None:
        super().__init__(transaction_iterator)

    def run(self, min_sup: float, min_conf: float) -> Tuple[List[FrequentPattern], List[AssociationRule]]:
        tree = FPTree(self.transactions, min_sup *
                      self.n_transactions, (None, 0))

        patterns = tree.mine_patterns(min_sup * self.n_transactions)
        rules = []
        for s, sup in patterns.items():
            for i in range(1, len(s)):
                for cond in comb(s, i):
                    cond = itemset(cond)
                    if cond in patterns:
                        r = itemset(s - cond)
                        lower_support = patterns[cond]
                        conf = sup / lower_support
                        if conf >= min_conf:
                            rules.append(AssociationRule(
                                cond, r, sup / self.n_transactions, conf))
        patterns = [
            FrequentPattern(k, v / self.n_transactions)
            for k, v in patterns.items()
        ]
        return patterns, rules


if __name__ == '__main__':
    # data = Transactions.parse(['ACD', 'BCE', 'ABCE', 'BE'])
    data = Transactions.parse(['MONKEY', 'DONKEY', 'MAKE', 'MUCKY', 'COOKIE'])
    # data = Transactions.parse(['ABCD', 'ABC', 'AB', 'A'])
    sol = FPGrowth(data)
    sol.run_and_pprint(0.5, 0.8)

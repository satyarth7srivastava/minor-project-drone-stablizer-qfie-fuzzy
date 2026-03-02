"""
fuzzy_partitions
================
Data structures for fuzzy linguistic variables and rule parsing.
Used by QFS.py for low-level quantum circuit construction.
"""

import math


class FuzzyPartition:
    """A linguistic variable split into named fuzzy sets.

    Parameters
    ----------
    name : str
        Name of the linguistic variable (e.g. 'temperature').
    fuzzy_sets : list[str]
        Ordered list of fuzzy-set labels (e.g. ['cold', 'warm', 'hot']).
    """

    def __init__(self, name, fuzzy_sets):
        self.name = name
        self.sets = list(fuzzy_sets)

    def len_partition(self):
        """Return the number of fuzzy sets in this partition."""
        return len(self.sets)

    def __repr__(self):
        return f"FuzzyPartition(name={self.name!r}, sets={self.sets})"


class fuzzy_rules:
    """Utility for parsing natural-language fuzzy-rule strings into
    tokenised lists compatible with QFS.convert_rule().

    The returned list alternates between:
        variable_name, binary_encoding, keyword ('and'/'then'), ...

    Binary encoding depends on the encoding scheme:
      - logarithmic: binary representation of the set index
        (width = ceil(log2(n_sets + 1)))
      - linear: one-hot string of length n_sets
    """

    def add_rules(self, rule_string, partitions, encoding='logaritmic'):
        """Parse *rule_string* using the given partitions.

        Parameters
        ----------
        rule_string : str
            e.g. 'if temp is hot and wind is strong then fan is high'
        partitions : list[FuzzyPartition]
            All partitions (inputs + output) so set names can be resolved.
        encoding : str
            'logaritmic' or 'linear'

        Returns
        -------
        list[str]
            Token list consumed by QFS.convert_rule().
        """
        partition_map = {p.name: p for p in partitions}
        tokens = rule_string.strip().split()
        result = []

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            # Skip keyword 'if'
            if tok == 'if':
                i += 1
                continue

            # Keywords that go straight into the result
            if tok in ('and', 'then'):
                result.append(tok)
                i += 1
                continue

            # 'not' keyword (for negation in linear encoding)
            if tok == 'not':
                result.append('not')
                i += 1
                continue

            # Variable name  →  expect  'is [not] <set_name>'
            if tok in partition_map:
                partition = partition_map[tok]
                result.append(tok)

                if i + 1 < len(tokens) and tokens[i + 1] == 'is':
                    offset = 2  # skip 'is'

                    # Optional 'not'
                    if (i + offset < len(tokens) and
                            tokens[i + offset] == 'not'):
                        result.append('not')
                        offset += 1

                    set_name = tokens[i + offset]
                    idx = partition.sets.index(set_name)

                    if encoding == 'logaritmic':
                        n_bits = math.ceil(
                            math.log(partition.len_partition() + 1, 2)
                        )
                        binary = format(idx, f'0{n_bits}b')
                    else:  # linear
                        n_bits = partition.len_partition()
                        bits = ['0'] * n_bits
                        bits[idx] = '1'
                        binary = ''.join(bits)

                    result.append(binary)
                    i += offset + 1
                    continue

            # Fallback — skip unknown tokens
            i += 1

        return result

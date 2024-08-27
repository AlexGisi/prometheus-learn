from typing import Optional
from os import path
import re
import numpy as np


def get_mirror():
    """Extend a 64-bit mirror to work on a vectorized board.

    Returns:
        list
    """
    mirror64 = [
        56, 57, 58, 59, 60, 61, 62, 63,
        48, 49, 50, 51, 52, 53, 54, 55,
        40, 41, 42, 43, 44, 45, 46, 47,
        32, 33, 34, 35, 36, 37, 38, 39,
        24, 25, 26, 27, 28, 29, 30, 31,
        16, 17, 18, 19, 20, 21, 22, 23,
        8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7
    ]
    mirror = [0 for _ in range(64*6)]
    for i in range(6):
        for j in range(64):
            mirror[i*64 + j] = mirror64[j] + i*64
    return mirror


def weights_from_file(fp: str, dtype) -> tuple[list[str], np.array]:
    with open(fp, 'r') as f:
        con = f.read()

    name_pattern = re.compile(r'(\w+Table)')
    name_matches = name_pattern.finditer(con)
    names = [match.group(1) for match in name_matches]

    weight_pattern = re.compile(r'([-0-9]+)\s*,')
    weight_matches = weight_pattern.finditer(con)
    weights = [int(match.group(1)) for match in weight_matches]

    assert len(weights) == (len(names) * 64)

    return names, np.array(weights, dtype=dtype)

def weights_to_file(names: list[str], weights: list[int], fp: Optional[str]) -> str:
    assert len(weights) == (len(names) * 64)

    out = ""
    for i, name in enumerate(names):
        out += "const int " + name + "[64] = {"
        for j in range(0, 64):
            if j % 8 == 0:
                out += '\n    '
            out += str(int(np.floor(weights[j + 64*i]))) + '\t,\t'
        out += '\n};\n'

        if i != len(names)-1:
            out += '\n'

    if fp:
        with open(fp, 'w') as f:
            f.write(out)

    return out


if __name__ == '__main__':
    n, w = weights_from_file(path.join(path.dirname(path.dirname(__file__)), 'io', 'weights_initial.txt'))
    s = weights_to_file(n, w, path.join(path.dirname(path.dirname(__file__)), 'io', 'weights_final.txt'))

from typing import Optional
from os import path
import re
import numpy as np


def weights_from_file(fp: str) -> tuple[list[str], np.array]:
    with open(fp, 'r') as f:
        con = f.read()
    
    name_pattern = re.compile(r'(\w+Table)')
    name_matches = name_pattern.finditer(con)
    names = [match.group(1) for match in name_matches]
    
    weight_pattern = re.compile(r'([-0-9]+)\s*,')
    weight_matches = weight_pattern.finditer(con)
    weights = [int(match.group(1)) for match in weight_matches]
    
    assert len(weights) == (len(names) * 64)
        
    return names, np.array(weights, dtype=np.float16)

def weights_to_file(names: list[str], weights: list[int], fp: Optional[str]) -> str:
    assert len(weights) == (len(names) * 64)
    
    out = ""
    for i, name in enumerate(names):
        out += "const int " + name + "[64] = {"
        for j in range(0, 64):
            if j % 8 == 0:
                out += '\n        ' 
            out += str(int(weights[j + 64*i])) + '\t\t,\t\t'   
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

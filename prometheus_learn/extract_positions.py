import re
import os
import csv

MAX_MATERIAL_DIFF = 0.0

FP_IN = "../data/out-eval-fens.pgn"
FP_OUT = "../data/balanced_positions.csv"

fp_in = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), FP_IN))
fp_out = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), FP_OUT)
)

print("Reading...")
with open(fp_in, "r") as f:
    string = f.read()

all_positions = []
game_strs = string.split("[Event")
print(f"Read {len(game_strs)} games. Compiling...")
for s in game_strs:
    fen_pattern = r"{ (\d*\.\d*) } { ([^}]*) }"  # Eval and fen (should be there after every move).
    fen_matches = re.findall(fen_pattern, s)

    res_pattern = r'"(\b(1-0|0-1|1\/2-1\/2))"'
    res_match = re.search(res_pattern, s)
    if res_match is None:
        continue  # Result can be "*" for abandoned games.

    res_str = res_match.groups()[0]
    if res_str == "1-0":
        res = 1
    elif res_str == "1/2-1/2":
        res = 0
    elif res_str == "0-1":
        res = -1
    else:
        raise AssertionError(f"Result string {res_str}")

    all_positions.extend(
        [{"result": res, "evaluation": float(m[0]), "fen": m[1]} for m in fen_matches]
    )

print(f"Compiled {len(all_positions)} positions total.")
print("Filtering...")

balanced = list(
    filter(lambda d: abs(d["evaluation"]) <= MAX_MATERIAL_DIFF, all_positions)
)

with open(fp_out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(balanced[0].keys()))

    writer.writeheader()
    writer.writerows(balanced)

print(f"Wrote {len(balanced)} pairs to {fp_out}")

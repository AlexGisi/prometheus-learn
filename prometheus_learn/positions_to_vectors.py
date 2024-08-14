import os
import subprocess
import concurrent.futures
import csv


PROMETHEUS_FP = "../bin/prometheus"
POSITIONS_FP = "../data/balanced_positions.csv"
OUT_FP = "../data/balanced_positions_with_vecs.csv"

prometheus_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), PROMETHEUS_FP)
)
positions_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), POSITIONS_FP)
)
out_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_FP)
)

print("Reading positions...")
with open(positions_fp, 'r', newline='') as f:
    r = csv.DictReader(f)
    positions = list(r)
    
print(f"Read {len(positions)} positions. Getting vectors...")

def get_vec(fen):
    command = [prometheus_fp, "fentovec", fen]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    output = result.stdout.strip().splitlines()

    if len(output) >= 2:
        return (output[0], output[1])
    else:
        raise ValueError(output)
    
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
    results = list(executor.map(get_vec, [p['fen'] for p in positions]))
    
print(f"Got {len(results)} vectors. Writing...")

for pos, vecs in zip(positions, results):
    pos['white'] = vecs[0]
    pos['black'] = vecs[1]
    
breakpoint()

with open(out_fp, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(positions[0].keys()))
    writer.writeheader()
    writer.writerows(positions)
    
print(f"Wrote to {out_fp}")

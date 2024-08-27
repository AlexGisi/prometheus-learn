import os
import subprocess
import concurrent.futures
import csv


THREAD_N = 10
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
with open(positions_fp, "r", newline="") as f:
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


BATCH_SIZE = 4096
total_positions = len(positions)
batches = [positions[i : i + BATCH_SIZE] for i in range(0, total_positions, BATCH_SIZE)]

for batch_num, batch in enumerate(batches):
    print(f"Processing batch {batch_num + 1}/{len(batches)}...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_N) as executor:
        results = list(executor.map(get_vec, [p["fen"] for p in batch]))

    for pos, vecs in zip(batch, results):
            pos["white"] = vecs[0]
            pos["black"] = vecs[1]
            
    with open(out_fp, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(positions[0].keys()))
        if batch_num == 0:
            writer.writeheader()
        writer.writerows(batch)
        f.flush()

print(f"Wrote to {out_fp}")

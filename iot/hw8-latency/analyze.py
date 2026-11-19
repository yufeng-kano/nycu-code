import csv
import statistics

import matplotlib.pyplot as plt

results = {}
for protocol in ['mqtt', 'http']:
    with open(f'latency_{protocol}.csv') as f:
        ms = [float(row[3]) * 1000 for row in csv.reader(f)]
    mean = statistics.mean(ms)
    std = statistics.stdev(ms)
    results[protocol] = ms
    print(f'{protocol.upper():4}  n={len(ms)}  mean={mean:.2f} ms  std={std:.2f} ms  CV={std / mean:.4f}'
          f'  median={statistics.median(ms):.1f}  min={min(ms):.1f}  max={max(ms):.1f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (protocol, ms) in zip(axes, results.items()):
    mean = statistics.mean(ms)
    cv = statistics.stdev(ms) / mean
    ax.hist(ms, bins=30, color='#4C78A8' if protocol == 'mqtt' else '#F58518', edgecolor='white')
    ax.axvline(mean, color='black', linestyle='--', linewidth=1)
    ax.set_title(f'{protocol.upper()} end-to-end delay (n={len(ms)}, mean={mean:.1f} ms, CV={cv:.3f})')
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Count')
fig.tight_layout()
fig.savefig('latency_histogram.png', dpi=150)
print('saved latency_histogram.png')

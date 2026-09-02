#!/usr/bin/env python3
"""Write the sustained telemetry summary into the FNO records.

The FNO harness launched tegrastats from the shell and never parsed it back, so
unlike WNO, Sp2GNO and DeepONet the FNO records carry no telemetry fields and the
published power, memory and thermal values could not be re-derived from the
record alone. This computes them from the co-located log over the 120 s
measurement window, matching the convention the other harnesses use, and adds
them under the same field names. Existing fields are not modified.
"""
import json, re, glob, os, datetime, statistics as st

D = 'families/fno/results/jetson_fno_unified_sustained'
TS = re.compile(r'^(\d\d)-(\d\d)-(\d{4}) (\d\d):(\d\d):(\d\d)')
VDD = re.compile(r'VDD_IN\s+(\d+)mW')
RAM = re.compile(r'RAM (\d+)/')
TMP = re.compile(r'\w+@([\d.]+)C')
WINDOW_S = 120

def parse(path):
    out = []
    for line in open(path, errors='replace'):
        m = TS.match(line)
        if not m:
            continue
        t = datetime.datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)),
                              int(m.group(4)), int(m.group(5)), int(m.group(6)))
        v = VDD.search(line); r = RAM.search(line)
        temps = [float(x) for x in TMP.findall(line)]
        out.append((t, int(v.group(1))/1000.0 if v else None,
                    int(r.group(1)) if r else None, max(temps) if temps else None))
    return out

n = 0
for j in sorted(glob.glob(D + '/*.json')):
    lg = j[:-5] + '_tegrastats.log'
    if not os.path.exists(lg):
        continue
    d = json.load(open(j))
    if 'vdd_in_mean_w' in d:
        continue
    rows = parse(lg)
    if not rows:
        continue
    end = rows[-1][0]
    win = [x for x in rows if x[0] >= end - datetime.timedelta(seconds=WINDOW_S)]
    W = [x[1] for x in win if x[1] is not None]
    R = [x[2] for x in win if x[2] is not None]
    T = [x[3] for x in win if x[3] is not None]
    mean_w = st.mean(W)
    d.update({
        'tegrastats_log': os.path.basename(lg),
        'tegrastats_samples': len(win),
        'telemetry_window_s': WINDOW_S,
        'vdd_in_mean_w': mean_w,
        'vdd_in_min_w': min(W),
        'vdd_in_max_w': max(W),
        'board_ram_mean_mb': st.mean(R),
        'board_ram_peak_mb': max(R),
        'peak_temp_c': max(T),
        'energy_j_per_inference': mean_w / d['throughput_inf_s'],
    })
    json.dump(d, open(j, 'w'), indent=2)
    n += 1
print(f'records updated: {n}')

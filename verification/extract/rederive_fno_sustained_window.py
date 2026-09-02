#!/usr/bin/env python3
"""Re-derive the FNO sustained telemetry in the aggregate cache over the 120 s
measurement window only, matching the convention the other families already use
and the values reported in the paper. Latency fields are untouched."""
import pickle, json, re, glob, os, datetime, statistics as st
import numpy as np

REPO = '/home/jetson/jjyoo3/neural-operator-jetson-benchmark'
D = REPO + '/families/fno/results/jetson_fno_unified_sustained'
TS = re.compile(r'^(\d\d)-(\d\d)-(\d{4}) (\d\d):(\d\d):(\d\d)')
VDD = re.compile(r'VDD_IN\s+(\d+)mW')

def win(path, seconds=120):
    rows = []
    for line in open(path, errors='replace'):
        m = TS.match(line)
        if not m: continue
        t = datetime.datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)),
                              int(m.group(4)), int(m.group(5)), int(m.group(6)))
        v = VDD.search(line)
        rows.append((t, int(v.group(1))/1000.0 if v else None))
    end = rows[-1][0]
    return [x[1] for x in rows if x[0] >= end - datetime.timedelta(seconds=seconds) and x[1] is not None]

P = REPO + '/verification/agg_all.pkl'
A = pickle.load(open(P, 'rb'))
changed = 0
for case, rec in A['fno_s'].items():
    W, P95, J = [], [], []
    ok = True
    for rep in (1, 2, 3):
        j, lg = f'{D}/{case}_rep{rep}.json', f'{D}/{case}_rep{rep}_tegrastats.log'
        if not (os.path.exists(j) and os.path.exists(lg)):
            ok = False; break
        thr = json.load(open(j))['throughput_inf_s']
        w = win(lg)
        W.append(st.mean(w)); P95.append(float(np.percentile(w, 95))); J.append(st.mean(w)/thr)
    if not ok: 
        print('  skip (records absent):', case); continue
    rec['W'], rec['p95W'], rec['J'] = st.mean(W), st.mean(P95), st.mean(J)
    changed += 1
pickle.dump(A, open(P, 'wb'))
print(f'FNO sustained entries re-derived to the 120 s window: {changed}')
for k in ['darcy_base_r85_fp32_strict','darcy_base_r141_fp32_strict','darcy_base_r281_fp32_strict','burgers_base_r2048_fp32_strict']:
    v = A['fno_s'][k]; print(f"  {k:34s} W={v['W']:.4f}  J={v['J']:.4f}")

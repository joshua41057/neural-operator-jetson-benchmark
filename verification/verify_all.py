#!/usr/bin/env python3
import os, pathlib
REPO  = pathlib.Path(__file__).resolve().parent.parent
PAPER = pathlib.Path(os.environ.get("PAPER_DIR", "")) if os.environ.get("PAPER_DIR") else None
if PAPER is None or not PAPER.exists():
    raise SystemExit(
        "verify_all.py checks the paper's tables against the measured records.\n"
        "Point it at the LaTeX sources:\n"
        "    PAPER_DIR=/path/to/overleaf_work python verify_all.py")
"""Independent verification of the Overleaf tree against the unified aggregates.

Deliberately does not import patch_tables.py: it re-parses every numeric cell out
of the .tex files and re-derives what that cell should be from agg_all.pkl, so a
bug in the patcher cannot hide behind itself. Also runs the cross-table and
physical-invariant checks that no single table can catch on its own.
"""
import pickle
import re
import sys

W = str(PAPER) + "/"
D = pickle.load(open(REPO / "verification" / "agg_all.pkl", "rb"))
FS, DS, WS, SS, HS, SR = (D["fno_s"], D["don_s"], D["wno_s"], D["sp_s"],
                          D["heat_s"], D["short"])
fails, checks = [], 0


def bad(msg):
    fails.append(msg)


def body(path, label, anchor=r"\midrule", end=r"\bottomrule"):
    s = open(W + path).read()
    i = s.index("\\label{%s}" % label)
    a = s.index(anchor, i) + len(anchor)
    return s[a:s.index(end, a)]


def cells(text):
    """Yield the &-split cells of each data row, skipping rules and headers."""
    for line in text.split("\\\\"):
        line = re.sub(r"\\(midrule|bottomrule|cmidrule\(lr\)\{[^}]*\}|"
                      r"multirow\{\d+\}\{\*\}\{[^}]*\})", "", line).strip()
        if not line or "&" not in line:
            continue
        yield [c.strip() for c in line.split("&")]


def eq(got, exp, nd, ctx):
    global checks
    checks += 1
    if got.replace("---", "--") == "--":
        return
    if got != f"{exp:.{nd}f}":
        bad(f"{ctx}: tex={got} 집계={exp:.{nd}f}")


# ---------------------------------------------------------------- Table 9
T9 = [("Burgers FP32", FS, "burgers_base_r2048_fp32_strict"),
      (r"Darcy 281$\times$281 FP32", FS, "darcy_base_r281_fp32_strict"),
      ("Burgers FP32", DS, "burgers_base_fp32_strict"),
      (r"Darcy 281$\times$281 FP32", DS, "darcy_r281_fp32_strict"),
      (r"Darcy 281$\times$281 FP16 Native", DS, "darcy_r281_fp16_native"),
      ("Burgers @2048 FP32", WS, "wno_burgers_base_r2048_fp32_strict"),
      (r"Darcy 281$\times$281 FP32", WS, "wno_darcy_base_r281_fp32_strict"),
      ("Burgers @2048 FP32", SS, "sp2gno_burgers_base_s2048_fp32_strict"),
      (r"Darcy $141\times141$ FP32", SS, "sp2gno_darcy_base_r141_fp32_strict"),
      ("Heat Exchanger (full) FP32", HS, "full_fp32"),
      ("Heat Exchanger (spectral-only) FP32", HS, "spectral_fp32"),
      ("Heat Exchanger (2-layer) FP32", HS, "layer2_fp32")]
rows = list(cells(body("sections/07_telemetry.tex", "tab:jetson_telemetry")))
if len(rows) != 12:
    bad(f"Table 9: 행 수 {len(rows)} != 12")
for r, (case, src, k) in zip(rows, T9):
    v = src[k]
    if r[1] != case:
        bad(f"Table 9: 케이스 순서 어긋남 {r[1]!r} != {case!r}")
    for got, exp, nd, cn in ((r[2], v["med"], 3, "med"), (r[3], v["W"], 3, "W"),
                             (r[4], v["J"], 4, "J"), (r[5], v["cuda"], 2, "cuda"),
                             (r[6], v["ram"], 0, "ram")):
        eq(got, exp, nd, f"Table 9 [{case}] {cn}")

# ------------------------------------------------- Tables 22/29/34/38 (sustained)
SUST = [("Table 22", "appendices/b_fno_supplementary.tex",
         "tab:appendix_fno_long_energy_full", FS, 10),
        ("Table 29", "appendices/c_deeponet_supplementary.tex",
         "tab:appendix_deeponet_long_energy_full", DS, 10),
        ("Table 34", "appendices/d_wno_supplementary.tex",
         "tab:appendix_wno_long_energy_full", WS, 9),
        ("Table 38", "appendices/e_sp2gno_supplementary.tex",
         "tab:appendix_sp2gno_long_energy_full", {**SS, **HS}, 9)]
for name, path, label, src, ncol in SUST:
    kw = dict(anchor=r"\endlastfoot", end=r"\end{longtable}") if ncol == 9 and \
        name == "Table 38" else {}
    rows = [r for r in cells(body(path, label, **kw)) if len(r) == ncol]
    # Every numeric cell must appear somewhere in the family's aggregate: match
    # each row by its median, then confirm the rest of the row agrees.
    lut = {}
    for k, v in src.items():
        lut.setdefault(f"{v['med']:.3f}", []).append((k, v))
    matched = 0
    for r in rows:
        if r[2].replace("---", "--") == "--":
            continue
        cand = lut.get(r[2])
        if not cand:
            bad(f"{name}: median {r[2]} 가 집계에 없음 ({r[0]} / {r[1]})")
            continue
        k, v = min(cand, key=lambda c: abs(c[1]["p95"] - float(r[3])))
        matched += 1
        off = 0 if ncol == 9 else 1
        eq(r[3], v["p95"], 3, f"{name} [{k}] p95")
        eq(r[4], v["thr"], 2, f"{name} [{k}] thr")
        eq(r[5], v["W"], 3, f"{name} [{k}] W")
        if ncol == 10:
            eq(r[6], v["p95W"], 3, f"{name} [{k}] p95W")
        eq(r[6 + off], v["J"], 4, f"{name} [{k}] J")
        eq(r[7 + off], v["ram"], 0, f"{name} [{k}] ram")
        eq(r[8 + off], v["temp"], 1, f"{name} [{k}] temp")
    print(f"  {name}: {len(rows)}행 중 {matched}행 수치 대조")

# ------------------------------------------- Tables 17/18/23/26/33/36 (short-run)
SHORT = [("Table 17", "appendices/b_fno_supplementary.tex",
          "tab:appendix_fno_fp32_full", "FNO", 8, 5),
         ("Table 18", "appendices/b_fno_supplementary.tex",
          "tab:appendix_fno_fp32_ablation_compact", "FNO_ABL", 7, 4),
         ("Table 26", "appendices/c_deeponet_supplementary.tex",
          "tab:appendix_deeponet_fp32_full", "DON", 8, 5),
         ("Table 33", "appendices/d_wno_supplementary.tex",
          "tab:appendix_wno_fp32_full", "WNO", 7, 4),
         ("Table 36", "appendices/e_sp2gno_supplementary.tex",
          "tab:appendix_sp2gno_fp32_full", "SP2", 7, 4)]
for name, path, label, fam, ncol, mi in SHORT:
    rows = [r for r in cells(body(path, label)) if len(r) == ncol]
    lut = {}
    for k, v in SR[fam].items():
        lut.setdefault(f"{v['med']:.3f}", []).append((k, v))
    matched = 0
    for r in rows:
        cand = lut.get(r[mi])
        if not cand:
            bad(f"{name}: median {r[mi]} 가 집계에 없음 ({r[0]})")
            continue
        k, v = min(cand, key=lambda c: abs(c[1]["p95"] - float(r[mi + 1])))
        matched += 1
        eq(r[mi + 1], v["p95"], 3, f"{name} [{k}] p95")
        eq(r[mi + 2], v["cuda"], 2, f"{name} [{k}] cuda")
    print(f"  {name}: {len(rows)}행 중 {matched}행 수치 대조")

# --------------------------------------------------- Table 23 (frontier + flags)
for r in cells(body("appendices/b_fno_supplementary.tex",
                    "tab:appendix_fno_soft_frontier")):
    if len(r) != 7:
        continue
    res = re.search(r"(\d+)\\times", r[1]).group(1)
    key = ("darcy_base_trained281" if "base" in r[0] else
           "darcy_large_trained141") + "_on_" + res
    v = SR["FNO_FRONT"][key]
    eq(r[2], v["med"], 3, f"Table 23 [{key}] med")
    eq(r[3], v["cuda"], 2, f"Table 23 [{key}] cuda")
    for got, thr in zip(r[4:7], (200, 500, 1000)):
        checks += 1
        want = "Yes" if v["med"] > thr else "No"
        if got != want:
            bad(f"Table 23 [{key}] {thr}ms 플래그: tex={got} 기대={want} "
                f"(med={v['med']:.3f})")

# ---------------------------------------------------------------- Table 5 / 11
# Table 5 now reads error and latency from the same (resolution-group) checkpoint.
T5 = [("FNO", 3.279), ("DeepONet", 1.669), ("WNO", 48.071), ("Sp$^2$GNO", 12.239),
      ("FNO", 6.685), ("DeepONet", 8.121), ("WNO", 69.746), ("Sp$^2$GNO", 81.932)]
rows = [r for r in cells(body("sections/05_fp32.tex", "tab:accuracy_cost"))
        if len(r) == 6]
if len(rows) != 8:
    bad(f"Table 5: 행 수 {len(rows)} != 8")
for r, (op, med) in zip(rows, T5):
    checks += 1
    if r[1] != op:
        bad(f"Table 5: 연산자 순서 {r[1]!r} != {op!r}")
    if r[4] != f"{med:.1f}":
        bad(f"Table 5 [{op}] median: tex={r[4]} 집계={med:.1f}")

# Table 5 오차 열: Darcy 는 공통 200샘플 재평가값 (darcy_common200_gate.json)
T5E = ["8.5", "1.9", "2.2", "5.3", "1.6", "7.1", "5.2", "9.7"]
for r, want in zip(rows, T5E):
    checks += 1
    g = re.search(r"([0-9.]+)\\times", r[3])
    if not g or g.group(1) != want:
        bad(f"Table 5 오차: tex={r[3]} 기대 계수={want}")


t11 = body("sections/07_telemetry.tex", "tab:deployment_frontiers")
LAD = {"FNO": ("FNO", [f"darcy_fno_base_r{r}_seed{s}_torchscript_fp32"
                       for r, s in (("85", 2), ("421", 1))]),
       "DeepONet": ("DON", [f"darcy_deeponet_base_r{r}_torchscript_fp32"
                            for r in (85, 421)]),
       "WNO": ("WNO", [f"wno_darcy_base_r{r}" for r in (85, 421)]),
       "Sp$^2$GNO": ("SP2", [f"sp2gno_darcy_base_r{r}_fp32_strict"
                             for r in (85, 421)])}
RAMK = {"FNO": ("fno_s", ["darcy_base_r85_fp32_strict", "darcy_base_r421_fp32_strict"]),
        "DeepONet": ("don_s", ["darcy_r141_fp32_strict", "darcy_r421_fp32_strict"]),
        "WNO": ("wno_s", ["wno_darcy_base_r85_fp32_strict", "wno_darcy_base_r421_fp32_strict"]),
        "Sp$^2$GNO": ("sp_s", ["sp2gno_darcy_base_r85_fp32_strict",
                               "sp2gno_darcy_base_r421_fp32_strict"])}
for op, (fam, ks) in LAD.items():
    seg = t11[t11.index(op + "\\tnote") if op + "\\tnote" in t11 else t11.index(op + " &"):][:700]
    lo, hi = (SR[fam][k] for k in ks)
    for v, tag in ((lo, "하단"), (hi, "상단")):
        checks += 1
        if f"{v['med']:.1f}" not in seg:
            bad(f"Table 11 [{op}] {tag} 지연 {v['med']:.1f} 미기재")
    for v, tag in ((lo, "하단"), (hi, "상단")):
        checks += 1
        if f"{v['cuda']:.1f}" not in seg:
            bad(f"Table 11 [{op}] {tag} peak CUDA {v['cuda']:.1f} 미기재")
    dk, rk = RAMK[op]
    A = {"fno_s": FS, "don_s": DS, "wno_s": WS, "sp_s": SS}[dk]
    for k, tag in zip(rk, ("하단", "상단")):
        checks += 1
        r = A[k]["ram"]
        if f"{r:.0f}" not in seg:
            bad(f"Table 11 [{op}] {tag} board RAM {r:.0f} 미기재 (실측 {r})")

# ---------------------------------------- physical invariant: sustained >= short-run
INV = [("FNO Burgers", SR["FNO"]["burgers_fno_base_seed3_torchscript_fp32"],
        FS["burgers_base_r2048_fp32_strict"]),
       ("FNO Darcy 281", SR["FNO"]["darcy_fno_base_r281_seed1_torchscript_fp32"],
        FS["darcy_base_r281_fp32_strict"]),
       ("DON Burgers", SR["DON"]["burgers_deeponet_base_torchscript_fp32"],
        DS["burgers_base_fp32_strict"]),
       ("WNO Burgers", SR["WNO"]["wno_burgers_base_r2048"],
        WS["wno_burgers_base_r2048_fp32_strict"]),
       ("WNO Darcy 421", SR["WNO"]["wno_darcy_base_r421"],
        WS["wno_darcy_base_r421_fp32_strict"]),
       ("SP2 Darcy 421", SR["SP2"]["sp2gno_darcy_base_r421_fp32_strict"],
        SS["sp2gno_darcy_base_r421_fp32_strict"])]
for name, s, l in INV:
    checks += 1
    if l["med"] < s["med"]:
        bad(f"물리 불변식 위반 [{name}]: sustained {l['med']:.3f} < "
            f"short-run {s['med']:.3f}")

# --------------------------------------------------- figure/table cross-agreement
import json
f2 = json.load(open(str(REPO / "figures") + "/fig2_full_data.json"))
# Fig 2 panels (a)/(b) keep the scale-group checkpoints.
for fam, k, exp in (("FNO", "Burgers", 3.175), ("DeepONet", "Burgers", 1.669),
                    ("WNO", "Burgers", 48.071), ("Sp2GNO", "Burgers", 12.239),
                    ("FNO", "Darcy", 6.661), ("DeepONet", "Darcy", 8.362),
                    ("WNO", "Darcy", 69.746), ("Sp2GNO", "Darcy", 81.932)):
    checks += 1
    if abs(f2[fam]["a"][k][0] - exp) > 1e-3:
        bad(f"Fig 2 [{fam}/{k}]: {f2[fam]['a'][k][0]} != Table 5 {exp}")

f3 = json.load(open(str(REPO / "figures") + "/fig3_precision_data.json"))
F3T9 = {("FNO", "Burgers 2048"): FS["burgers_base_r2048_fp32_strict"],
        ("DeepONet", "Burgers 2048"): DS["burgers_base_fp32_strict"],
        ("WNO", "Burgers 2048"): WS["wno_burgers_base_r2048_fp32_strict"],
        ("Sp2GNO", "Burgers 2048"): SS["sp2gno_burgers_base_s2048_fp32_strict"],
        ("Sp2GNO", "Darcy 141²"): SS["sp2gno_darcy_base_r141_fp32_strict"]}
for b in f3:
    key = (b["family"], b["workload"])
    if key in F3T9:
        checks += 1
        if abs(b["cells"]["FP32"][0] - F3T9[key]["med"]) > 5e-4:
            bad(f"Fig 3 [{key}] FP32: {b['cells']['FP32'][0]} != "
                f"Table 9 {F3T9[key]['med']:.3f}")

# ------------------------------------------------- every cell has R = 3 behind it
for tab, name in ((FS, "fno_s"), (DS, "don_s"), (WS, "wno_s"), (SS, "sp_s"),
                  (HS, "heat_s")):
    for k, v in tab.items():
        checks += 1
        if v["n"] != 3:
            bad(f"반복 수 부족 [{name}/{k}]: n={v['n']}")
for fam, tab in SR.items():
    for k, v in tab.items():
        checks += 1
        if v["n"] != 3:
            bad(f"반복 수 부족 [short/{fam}/{k}]: n={v['n']}")

print(f"\n검사 {checks}건 수행")
if fails:
    print(f"불일치 {len(fails)}건:")
    for f in fails:
        print("   " + f)
    sys.exit(1)
print("불일치 0건")

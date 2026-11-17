#!/usr/bin/env python3
"""Compare regenerated tables with reference copies. CSV and whitespace tables are compared as multisets of rows
(row order follows the file system and may differ), numbers to a relative tolerance; other files byte for byte.
Usage: compare.py <reference dir> <file>... [--rtol 1e-9]. Exit 1 if anything differs."""
import csv, math, os, sys
from collections import Counter
args = sys.argv[1:]; rtol = 1e-9
if "--rtol" in args:
    i = args.index("--rtol"); rtol = float(args[i + 1]); del args[i:i + 2]
ref, files = args[0], args[1:]

def norm(cell):
    try:
        v = float(cell)
        return "nan" if math.isnan(v) else f"{v:.12g}" if rtol <= 1e-9 else round(v, max(0, int(-math.log10(rtol))))
    except ValueError:
        return cell

def rows(p):
    if p.endswith(".csv"):
        r = list(csv.reader(open(p)))
    elif p.endswith(".dat"):
        r = [l.split() for l in open(p) if not l.startswith("#")]
    else:
        return None
    return Counter(tuple(norm(c) for c in row) for row in r)

bad = 0
for f in files:
    a, b = os.path.join(ref, f), f
    if not os.path.exists(a):
        print(f"NEW   {f}"); continue
    ra, rb = rows(a), rows(b)
    same = (ra == rb) if ra is not None else open(a, "rb").read() == open(b, "rb").read()
    if not same:
        bad += 1
        extra = "" if ra is None else f" ({sum((ra - rb).values())} rows only in the reference, {sum((rb - ra).values())} only in the new file)"
        print(f"DIFF  {f}{extra}")
print(f"{len(files) - bad}/{len(files)} files reproduce the shipped versions" + (f" (rtol {rtol})" if rtol > 1e-9 else ""))
sys.exit(1 if bad else 0)

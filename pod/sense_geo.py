"""sense_geo.py OUT_CWS OUT_ACWS : decide the spiral sense from the two short fits' snap reports.

306B's sense_geo.py was not published; this stand-in uses what snap_meshes.py writes (snap/snap_report.tsv per winding:
frac_with_sheet, frac_snapped, median_abs_off, p90_abs_off, frac_rejected). The fit whose windings sit on the predicted
sheets has a higher snapped fraction, a smaller median offset and fewer sheet-jump rejections. Decision: a score per sense
= mean(frac_snapped) - mean(frac_rejected) - mean(median_abs_off) / 15; prints CW or ACW on stdout when the two scores differ
by more than 0.05 AND the winner is better on at least two of the three terms; prints nothing (not decisive) otherwise, so the
caller falls back to its tie-break. Details go to stderr.
"""
import csv, sys, statistics as st

def load(d):
    rows = list(csv.DictReader(open(d + '/snap/snap_report.tsv'), delimiter='\t'))
    if not rows:
        raise SystemExit(f'{d}: empty snap report')
    f = lambda k: st.fmean(float(r[k]) for r in rows)
    return dict(n=len(rows), snapped=f('frac_snapped'), rejected=f('frac_rejected'), off=f('median_abs_off'), sheet=f('frac_with_sheet'))

cw, acw = load(sys.argv[1]), load(sys.argv[2])
score = lambda m: m['snapped'] - m['rejected'] - m['off'] / 15.0
s_cw, s_acw = score(cw), score(acw)
better = {'CW': 0, 'ACW': 0}
for k, hi in (('snapped', True), ('rejected', False), ('off', False)):
    w = 'CW' if ((cw[k] > acw[k]) == hi) else 'ACW'
    if cw[k] != acw[k]: better[w] += 1
print(f'CW  n={cw["n"]} snapped={cw["snapped"]:.3f} rejected={cw["rejected"]:.3f} off={cw["off"]:.2f} sheet={cw["sheet"]:.3f} score={s_cw:.3f}', file=sys.stderr)
print(f'ACW n={acw["n"]} snapped={acw["snapped"]:.3f} rejected={acw["rejected"]:.3f} off={acw["off"]:.2f} sheet={acw["sheet"]:.3f} score={s_acw:.3f}', file=sys.stderr)
win = 'CW' if s_cw > s_acw else 'ACW'
if abs(s_cw - s_acw) > 0.05 and better[win] >= 2:
    print(f'decisive: {win} (terms won {better[win]}/3, margin {abs(s_cw - s_acw):.3f})', file=sys.stderr)
    print(win)
else:
    print(f'not decisive (margin {abs(s_cw - s_acw):.3f}, terms CW {better["CW"]} ACW {better["ACW"]})', file=sys.stderr)

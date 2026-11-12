# Reader V3 calibration of the 306B triage

positives: 17 text windows {'c0841': 5, 'c1447': 8, 'c306': 4}; nulls: 506 PHerc. 0800 windows

| feature | AUC vs 0800 nulls | null p99 | text recall above null p99 | null false alarms at 100 % text recall | at 90 % |
|---|---|---|---|---|---|
| score | 0.716 | 7.421 | 0.118 | 0.955 | 0.751 |
| glyph | 0.853 | 12.831 | 0.294 | 0.634 | 0.433 |
| shape | 0.715 | 0.643 | 0.0 | 0.803 | 0.605 |
| rel_tex | 0.504 | 1.199 | 0.0 | 0.937 | 0.818 |
| row_off | 0.767 | 0.0 | 0.0 | 0.97 | 0.549 |
| fib_off | 0.672 | 1.2 | 0.0 | 0.877 | 0.877 |
| crack | 0.526 | -0.0 | 0.0 | 0.873 | 0.809 |

shipped admission (score >= 4 or glyph >= 9.06): text 0.941, nulls 0.613
shipped REVIEW queue: text 0.588, nulls 0.099

line_find nulls: max 8.942, p99 7.496

- c1447_w058:fwd: best 3.875, best on text 1.316, chains 7, text face False
- c1447_w058:rev: best 9.046, best on text 9.046, chains 13, text face True
- c1447_w060:fwd: best 6.291, best on text 6.291, chains 8, text face False
- c1447_w060:rev: best 6.489, best on text 5.575, chains 12, text face True

text face per surface (pixel AUC of V3 d0 against the labels):
- c1447_w058: rev 0.85
- c1447_w060: rev 0.78
- c0841_w00: rev 0.917
- c0841_ag144: rev 0.824
- c0841_ag174: rev 0.849
- c306_claim: fwd 0.516

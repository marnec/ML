def fixtex(s):
 return s.replace('tabular', 'bmatrix').replace('{lr}', '').replace('\\toprule', '').replace('\\midrule', '').replace('\\bottomrule', '').replace('{r}', '').replace('\n', '')
 
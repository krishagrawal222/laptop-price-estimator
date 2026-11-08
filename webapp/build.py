import json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
d = pd.read_csv('../data/laptop_price_inr_cleaned.csv'); d['Other_GB'] = d.Flash_GB + d.Hybrid_GB
CAT = ['Company','TypeName','Cpu_Series','Gpu_Brand','OS_Family']
NUM = ['Inches','Screen_Width','Screen_Height','PPI','Touchscreen','IPS_Panel','Cpu_GHz','Ram_GB','SSD_GB','HDD_GB','Other_GB','Weight_kg']
KEY = ['co','ty','cpu','gpu','os']
def pipe():
    pre = ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT)], remainder='passthrough', sparse_threshold=0)
    return Pipeline([('p', pre), ('f', RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1))])
X, y = d[CAT+NUM], d.Price_INR
Xa, Xb, ya, yb = train_test_split(X, y, test_size=.2, random_state=42)
chk = pipe().fit(Xa, ya); pr = chk.predict(Xb); cut = lambda p: np.where(p < 60000, 0, np.where(p <= 120000, 1, 2))
metrics = dict(r2=round(r2_score(yb, pr), 3), mae=int(mean_absolute_error(yb, pr)), cat_acc=round(float((cut(pr) == cut(yb)).mean()*100), 1), test=len(yb))
m = pipe().fit(X, y); enc = m.named_steps['p'].named_transformers_['c']
trees = []
for e in m.named_steps['f'].estimators_:
    t = e.tree_; leaf = t.children_left == -1
    trees.append(dict(f=[-1 if l else int(f) for f, l in zip(t.feature, leaf)], t=[round(float(a), 6) for a in t.threshold],
                      l=t.children_left.tolist(), r=t.children_right.tolist(), v=[int(round(v[0][0])) for v in t.value]))
model = dict(cats=['co','ty','cpu','gpu','os'], values={k: list(c) for k, c in zip(['co','ty','cpu','gpu','os'], enc.categories_)}, trees=trees, metrics=metrics)
# parity check: exported trees must reproduce sklearn's predictions
Xt = m.named_steps['p'].transform(X.iloc[:200]); mine = []
for row in Xt:
    s = 0
    for t in trees:
        i = 0
        while t['f'][i] >= 0: i = t['l'][i] if row[t['f'][i]] <= t['t'][i] else t['r'][i]
        s += t['v'][i]
    mine.append(s/len(trees))
print('max parity gap (INR):', np.abs(np.array(mine) - m.predict(X.iloc[:200])).max(), metrics)
rows = [[r.Company, r.Product, r.TypeName, r.Cpu_Series, r.Gpu_Brand, r.OS_Family, float(r.Inches), int(r.Screen_Width), int(r.Screen_Height), int(r.Touchscreen), int(r.IPS_Panel), float(r.Cpu_GHz), int(r.Ram_GB), int(r.SSD_GB), int(r.HDD_GB), int(r.Other_GB), float(r.Weight_kg), int(r.Price_INR), r.Price_Category] for r in d.itertuples()]
html = open('template.html').read().replace('/*DATA*/null', json.dumps(rows, ensure_ascii=False, separators=(',', ':'))).replace('/*MODEL*/null', json.dumps(model, separators=(',', ':')))
import os; os.makedirs('..', exist_ok=True)
open('../index.html', 'w').write(html); print(round(len(html)/1e6, 2), 'MB')

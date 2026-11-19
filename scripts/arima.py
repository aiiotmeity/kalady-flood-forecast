import sys,os;sys.path.insert(0,os.path.dirname(__file__))
import numpy as np,pandas as pd,pipe,json
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
z=np.load('./data/win_base.npz');Xp,Y=z['Xp'],z['Y'];s=pipe.split(len(Xp))
hw=pipe.load_raw('./data/lime.xlsx');lv=hw.level.values
ntr=s+pipe.PAST_HOURS   # rows available before first validation origin
d=np.diff(lv[:ntr])
sel=ar_select_order(d,maxlag=26,ic='aic',trend='c',glob=False)
p=max(sel.ar_lags) if sel.ar_lags else 1
res=AutoReg(d,lags=p,trend='c').fit()
c=res.params[0];phi=res.params[1:]
print('ARIMA(%d,1,0) AIC %.1f'%(p,res.aic))
dall=np.diff(lv)
n=len(Xp);pred=np.zeros((n-s,6))
for j,i in enumerate(range(s+pipe.PAST_HOURS,n+pipe.PAST_HOURS)):
    hist=list(dall[i-1-p:i-1])  # diffs up to level[i-1]-level[i-2]
    L=lv[i-1]
    for h in range(6):
        nd=c+np.dot(phi,hist[::-1][:p]); hist.append(nd); L=L+nd; pred[j,h]=L
np.save('./runs_arima.npy',pred)
json.dump({'p':int(p),'aic':float(res.aic)},open('./arima.json','w'))
yv=Y[s:,:,0]
for h in range(6):
    e=yv[:,h]-pred[:,h];print('H+%d MAE %.4f RMSE %.4f'%(h+1,np.abs(e).mean(),np.sqrt((e**2).mean())))

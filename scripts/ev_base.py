import sys,os;sys.path.insert(0,os.path.dirname(__file__))
import numpy as np,pandas as pd,pipe,json
from statsmodels.tsa.ar_model import AutoReg
z=np.load('./data/win_base.npz');Xp,Y,t=z['Xp'],z['Y'],pd.DatetimeIndex(z['t']);s=pipe.split(len(Xp))
ev=(t>='2018-08-01')&(t<='2018-08-31 23:00'); idx=np.where(ev)[0]
hw=pipe.load_raw('./data/lime.xlsx');lv=hw.level.values;hidx=hw.index
p=json.load(open('./arima.json'))['p']
# fit on training rows excluding the buffer period around the event
ntr=s+pipe.PAST_HOURS; d=np.diff(lv[:ntr]); dt=hidx[1:ntr]
keep=~((dt>='2018-07-30')&(dt<='2018-09-02 23:00'))
# build lagged design manually excluding rows whose target or lags fall in buffer
Xd=np.column_stack([d[p-k-1:len(d)-k-1] for k in range(p)]); yd=d[p:]; kk=keep[p:].copy()
for k in range(p): kk&=keep[p-k-1:len(d)-k-1]
A=np.column_stack([np.ones(kk.sum()),Xd[kk]]); coef=np.linalg.lstsq(A,yd[kk],rcond=None)[0]; c,phi=coef[0],coef[1:]
dall=np.diff(lv); pa=np.zeros((len(idx),6))
for j,w in enumerate(idx):
    i=w+pipe.PAST_HOURS; hist=list(dall[i-1-p:i-1]); L=lv[i-1]
    for h in range(6):
        nd=c+np.dot(phi,hist[::-1][:p]); hist.append(nd); L+=nd; pa[j,h]=L
per=np.repeat(Xp[idx,-1,0][:,None],6,1)
np.save('./ev_arima.npy',pa); np.save('./ev_persist.npy',per); np.save('./ev_y.npy',Y[idx,:,0])
yv=Y[idx,:,0]
for n,q in [('persist',per),('arima',pa)]:
    e=yv-q;print(n,'MAE',np.abs(e).mean(0).round(3),'RMSE',np.sqrt((e**2).mean(0)).round(3))
print('n',len(idx),'max level',yv.max(),'hours>8',(yv[:,0]>8).sum(),'>11',(yv[:,0]>11).sum())

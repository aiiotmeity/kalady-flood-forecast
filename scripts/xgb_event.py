import sys,os;sys.path.insert(0,os.path.dirname(__file__))
import numpy as np,pandas as pd,pipe,xgboost as xgb,json,time
seed=int(sys.argv[1]);t0=time.time()
z=np.load('./data/win_base.npz');Xp,Xf,Y,t=z['Xp'],z['Xf'],z['Y'],pd.DatetimeIndex(z['t']);s=pipe.split(len(Xp))
ev=(t>='2018-08-01')&(t<='2018-08-31 23:00');buf=(t>='2018-07-30')&(t<='2018-09-02')
tr=np.where((np.arange(len(Xp))<s)&~buf)[0]
X=np.hstack([Xp.reshape(len(Xp),-1),Xf.reshape(len(Xf),-1)]).astype('float32');last=Xp[:,-1,0]
pe=np.zeros((ev.sum(),6))
for h in range(6):
    y=Y[:,h,0]-last
    m=xgb.XGBRegressor(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.5,tree_method='hist',random_state=seed,n_jobs=2,early_stopping_rounds=30)
    m.fit(X[tr],y[tr],eval_set=[(X[s:],y[s:])],verbose=False); pe[:,h]=m.predict(X[ev])+last[ev]; print(h,time.time()-t0,flush=True)
np.save(f'./xgb_event_s{seed}.npy',pe)
yv=Y[ev,:,0];e=yv-pe;hi=yv[:,0]>8
print(json.dumps({'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt((e**2).mean())),'mae8':float(np.abs(e[hi]).mean()),'rmse8':float(np.sqrt((e[hi]**2).mean())),'rmse_h6':float(np.sqrt((e[:,5]**2).mean()))}))

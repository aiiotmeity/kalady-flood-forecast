import sys,os;sys.path.insert(0,os.path.dirname(__file__))
import numpy as np,pipe,xgboost as xgb,time,json
seed=int(sys.argv[1]); out=f'./runs/xgb_s{seed}'; os.makedirs(out,exist_ok=True)
if os.path.exists(out+'/pred.npy'): sys.exit()
z=np.load('./data/win_base.npz');Xp,Xf,Y=z['Xp'],z['Xf'],z['Y'];s=pipe.split(len(Xp))
X=np.hstack([Xp.reshape(len(Xp),-1),Xf.reshape(len(Xf),-1)]).astype('float32');last=Xp[:,-1,0]
pred=np.zeros((len(X)-s,6));t0=time.time();info={}
for h in range(6):
    y=Y[:,h,0]-last
    m=xgb.XGBRegressor(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.5,
                       tree_method='hist',random_state=seed,n_jobs=int(os.environ.get('NT',4)),device=os.environ.get('XGB_DEVICE','cpu'),early_stopping_rounds=30)
    m.fit(X[:s],y[:s],eval_set=[(X[s:],y[s:])],verbose=False)
    pred[:,h]=m.predict(X[s:])+last[s:]; info[h]=int(m.best_iteration)
np.save(out+'/pred.npy',pred);json.dump({'best_iter':info,'seconds':time.time()-t0},open(out+'/info.json','w'))
print('xgb',seed,time.time()-t0)

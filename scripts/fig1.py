import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9})
fig,ax=plt.subplots(figsize=(7.2,5.4)); ax.set_xlim(0,100); ax.set_ylim(-1,72); ax.axis('off')
def band(y0,h,label):
    ax.add_patch(FancyBboxPatch((6,y0),93,h,boxstyle='round,pad=0,rounding_size=0.8',fc='#efefef',ec='none'))
    ax.text(3.2,y0+h/2,label,rotation=90,ha='center',va='center',style='italic',color='#555',fontsize=8.5,linespacing=0.9)
def box(x,y,w,h,t,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.8',fc='white',ec='#333',lw=1))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8,fontweight='bold' if bold else 'normal')
def arr(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color='#333',lw=1))
band(57,14,'Sensing'); band(44,12,'Data'); band(18,24,'Modelling'); band(0,17,'Decision\nsupport')
xs=[9,31,53,75]; labs=['Radar water level\nsensor, Kalady','Rain gauge\nIllithode','Rain gauge\nKuttampuzha','AWS-1\nASIET campus']
for x,l in zip(xs,labs): box(x,59.5,20,9,l)
box(10,46,37,8,'Cloud ingestion and\nobject storage (S3)'); box(58,46,39,8,'Historical record\n(India-WRIS)')
for x in [19,41]: arr(x,59.5,x,54.3)
for x in [63,85]: arr(x,59.5,x,54.3)
box(10,32,24,8,'Hourly alignment\nand gap filling'); box(38,32,24,8,'Unit-hydrograph\nrouting'); box(66,32,31,8,'Feature construction\n(24 × 39)')
arr(28,46,28,40.3); arr(78,46,78,40.3); arr(34,36,37.7,36); arr(62,36,65.7,36)
box(19,20,30,8,'Encoder-decoder LSTM\n(level-change output)',bold=True); box(56,20,32,8,'Masked future rainfall\ncovariate (6 × 2)')
arr(80,32,76,28.3); arr(56,24,49.3,24)
box(10,8,24,7,'6-hour level\nforecast'); box(38,8,24,7,'LIME attribution\nper lead time'); box(66,8,31,7,'GIS flood extent\nand web dashboard')
arr(30,20,24,15.3); arr(34,11.5,37.7,11.5); arr(62,11.5,65.7,11.5)
box(28,1.2,46,5.5,'District Disaster Management Authority',bold=True); arr(80,8,74.3,5.5)
plt.savefig('figs/fig1.png',dpi=300,bbox_inches='tight',pad_inches=0.05)

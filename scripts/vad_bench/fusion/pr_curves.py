from analyze import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
allidx=np.flatnonzero(sp); rows=[]
def pr(arr): c=arr[:,allidx,:4].sum(1); return c[:,0]/(c[:,0]+c[:,1]),c[:,0]/(c[:,0]+c[:,2])
def front(P,Rr):
    o=np.argsort(-Rr); best=-1; keep=[]
    for i in o:
        if P[i]>best+1e-9: keep.append(i); best=P[i]
    return keep
fig,axs=plt.subplots(1,2,figsize=(13,5.5))
for ax,h in zip(axs,HEADS):
    for r_,lab,st in (("silero","Silero thr sweep","k-"),("head",f"{h} head thr sweep","r-"),("or","OR grid frontier","g--"),("and","AND grid frontier","c--"),("mean","mean (w .25/.5/.75) frontier","m--"),("max","max sweep","y:"),("two","two-stage frontier","b-")):
        ps,arr=cand(r_,h); P,Rr=pr(arr); k=front(P,Rr); ax.plot(100*Rr[k],100*P[k],st,label=lab,lw=1.6)
        for i in k: rows.append((h,r_,str(ps[i]),round(100*P[i],2),round(100*Rr[i],2)))
    # LR out-of-fold curves
    for kind,st in (("lr6","orange"),("hgb6","brown"),("lr6nz","olive")):
        P=[];Rr=[]
        for ti in range(17):
            c=np.zeros(4)
            for f in range(4): o,_=LR[(h,kind,"f%d"%f)]; te=np.flatnonzero(sp&(fold==f)); c+=o[ti][te,:4].sum(0)
            P.append(c[0]/(c[0]+c[1])); Rr.append(c[0]/(c[0]+c[2]))
        P=np.array(P);Rr=np.array(Rr); ax.plot(100*Rr,100*P,"-",color=st,label=f"{kind} out-of-fold thr sweep",lw=1.6)
        for ti in range(17): rows.append((h,kind+"_oof",f"thr={0.1+0.05*ti:.2f}",round(100*P[ti],2),round(100*Rr[ti],2)))
    ax.set_xlim(80,100); ax.set_ylim(93,100); ax.set_xlabel("frame recall %"); ax.set_ylabel("frame precision %"); ax.set_title(f"Silero + {h} head (342 synthetic speech clips)"); ax.grid(alpha=.3); ax.legend(fontsize=7,loc="lower left")
plt.tight_layout(); plt.savefig("pr_curves.png",dpi=130)
import csv; w=csv.writer(open("pr_points.csv","w")); w.writerow(["head","rule","params","precision_pct","recall_pct"]); w.writerows(rows)

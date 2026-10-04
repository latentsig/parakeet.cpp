import numpy as np,torch
from vadlib import *; import corpus
# fixed normalisation statistics: (a) mean of per-file stats over 40 clean fusion-style clips; (b) raw LibriSpeech utterances (no pauses)
cl=corpus.clean_clips()
ms=[];ss=[]
for y,_ in cl:
    m=raw_logmel(y)[:-1].double(); ms.append(m.mean(0)); ss.append(m.std(0)) 
statA=(torch.stack(ms).mean(0).float(),torch.stack(ss).mean(0).float())
allm=torch.cat([raw_logmel(u.astype(np.float32))[:-1] for u in corpus.libri[:120]]).double()
statB=(allm.mean(0).float(),allm.std(0).float())
np.savez('fixed_stats.npz',aM=statA[0].numpy(),aS=statA[1].numpy(),bM=statB[0].numpy(),bS=statB[1].numpy())
print(statA[0][:5],statA[1][:5],statB[0][:5],statB[1][:5])

import numpy as np,soundfile as sf,json
from sig import *
x,sr=sf.read('talk/JaneMcGonigal-merged.wav',dtype='float32'); assert sr==16000
ex=x[60*SR:300*SR]                                  # 4 min excerpt
# find the quietest 1 s window between 100 and 140 s of the excerpt to cut at (a natural pause)
fr=len(ex)//1600; e=10*np.log10((ex[:fr*1600].reshape(fr,1600)**2).mean(1)+1e-12)
c=[(e[i:i+10].mean(),i) for i in range(1000,1400)]; _,i0=min(c); cut=(i0+5)*1600     # centre of that window
speech_rms_db=20*np.log10(rms(ex)); print('excerpt rms dBFS',speech_rms_db,'cut at',cut/SR,'s')
r=np.random.default_rng(4); L=60*SR
gens={'white':lambda: white(L,r),'pink':lambda: colored(L,r,1),'clicks':lambda: clicks(L,r,3,0),'music':lambda: music(L,r,0)}
meta={'cut':cut/SR,'ins':[cut/SR,cut/SR+60],'rms_db':speech_rms_db,'files':[]}
sf.write('talk/d/base.wav',ex,16000,subtype='PCM_16'); meta['files'].append(('base',None,None))
for tn,g in gens.items():
    n0=g()
    for rel in [-35,-20,-5]:
        n=at_db(n0,speech_rms_db+rel) if tn!='clicks' else (n0/ (np.abs(n0).max()+1e-9) * 10**((speech_rms_db+rel+14)/20)).astype(np.float32)  # clicks: set RMS ~ rel via crest ~ calibrate below
        if tn=='clicks': n=at_db(n0,speech_rms_db+rel)
        y=np.concatenate([ex[:cut],np.clip(n,-1,1),ex[cut:]]).astype(np.float32)
        nm=f'{tn}_{rel}'; sf.write(f'talk/d/{nm}.wav',y,16000,subtype='PCM_16'); meta['files'].append((nm,tn,rel))
json.dump(meta,open('talk/d/meta.json','w'))
print(len(meta['files']),'files')

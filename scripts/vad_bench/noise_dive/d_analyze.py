import json,os
meta=json.load(open('talk/d/meta.json')); a,b=meta['ins']
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def words_in(d,lo,hi): return [w for w in d['words'] if lo<=w['start']<hi]
rows=[]
print('Inserted 60 s block at %.1f-%.1f s of a 300 s talk excerpt. Words whose start lies inside the block = hallucinated (the inserted audio has no speech).'%(a,b))
for m in ['ultra','redux']:
    base=load(f'talk/out/base.{m}.vad0.json'); nb=len(base['words']) if base else None
    print(f'\n### {m} (talk excerpt alone: {nb} words without VAD, {len(load(f"talk/out/base.{m}.vad1.json")["words"]) if load(f"talk/out/base.{m}.vad1.json") else "-"} with VAD)')
    print('| inserted audio | level re talk RMS | words in block, no VAD | words in block, --vad | talk words outside block, no VAD / --vad (base %s) | VAD segments touching block (s covered / count) |'%nb)
    print('|---|---|---|---|---|---|')
    for f,tn,rel in meta['files']:
        if f=='base': continue
        d0=load(f'talk/out/{f}.{m}.vad0.json'); d1=load(f'talk/out/{f}.{m}.vad1.json'); sg=load(f'talk/out/{f}.{m}.segments.json')
        if not d1 or not sg: continue
        w0=words_in(d0,a,b) if d0 else None; w1=words_in(d1,a,b)
        out0=(len(d0['words'])-len(w0)) if d0 else None; out1=len(d1['words'])-len(w1)
        cov=sum(max(0,min(s['end'],b)-max(s['start'],a)) for s in sg['segments']); cnt=sum(1 for s in sg['segments'] if s['end']>a and s['start']<b)
        rows.append(dict(model=m,file=f,type=tn,rel=rel,w_novad=None if w0 is None else len(w0),w_vad=len(w1),cov=cov,cnt=cnt,sample1=' '.join(w['w'] for w in w1[:12]),sample0=' '.join(w['w'] for w in w0[:12]) if w0 else ''))
        print(f'| {tn} | {rel} dB | {"-" if w0 is None else len(w0)} | {len(w1)} | {"-" if out0 is None else out0} / {out1} | {cov:.0f} s / {cnt} |')
json.dump(rows,open('res_D.json','w'),indent=1)
print('\nexample hallucinated text:')
for r in rows:
    if r['w_vad'] or (r['w_novad'] or 0)>0: print(r['model'],r['file'],'| novad:',r['sample0'][:90],'| vad:',r['sample1'][:90])

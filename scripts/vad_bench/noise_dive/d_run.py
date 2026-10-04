import json,subprocess,sys,os,time
CLI=os.environ.get('PARAKEET_CLI','build/examples/cli/parakeet-cli'); S=os.environ.get('GGUF_ROOT','..')
M={'ultra':S+'/gguf/ultra-q8_0.gguf','redux':S+'/gguf/redux-keep.gguf'}
meta=json.load(open('talk/d/meta.json')); os.makedirs('talk/out',exist_ok=True)
files=[f[0] for f in meta['files']]
for m in ['ultra','redux']:
    for f in files:
        wav=f'talk/d/{f}.wav'
        o=f'talk/out/{f}.{m}.segments.json'
        if not os.path.exists(o):
            r=subprocess.run([CLI,'vad','--model',M[m],'--input',wav,'--mode','segments','--threads','4'],capture_output=True,text=True); open(o,'w').write(r.stdout)
        for vad in [1,0]:
            o=f'talk/out/{f}.{m}.vad{vad}.json'
            if os.path.exists(o): continue
            t=time.time()
            r=subprocess.run([CLI,'transcribe','--model',M[m],'--input',wav,'--json','--timestamps','--threads','4']+(['--vad'] if vad else []),capture_output=True,text=True)
            open(o,'w').write(r.stdout); print(m,f,vad,round(time.time()-t,1),r.returncode,len(r.stdout),flush=True)

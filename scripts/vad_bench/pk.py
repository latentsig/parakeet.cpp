"""ctypes wrapper over libparakeet for the VAD benchmarks. Set PARAKEET_LIB to the path of libparakeet.so."""
import ctypes, json, os, numpy as np
LIB=os.environ.get("PARAKEET_LIB","build/libparakeet.so")
class PK:
    def __init__(self, gguf):
        L=self.L=ctypes.CDLL(LIB)
        L.parakeet_capi_load.restype=ctypes.c_void_p; L.parakeet_capi_load.argtypes=[ctypes.c_char_p]
        for n in ("parakeet_capi_vad_pcm_json","parakeet_capi_vad_path_json","parakeet_capi_transcribe_pcm_batch_json"):
            getattr(L,n).restype=ctypes.c_void_p
        L.parakeet_capi_vad_pcm_json.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_char_p]
        L.parakeet_capi_vad_path_json.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_char_p]
        L.parakeet_capi_transcribe_pcm_batch_json.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        L.parakeet_capi_free_string.argtypes=[ctypes.c_void_p]
        L.parakeet_capi_last_error.restype=ctypes.c_char_p; L.parakeet_capi_last_error.argtypes=[ctypes.c_void_p]
        self.ctx=L.parakeet_capi_load(gguf.encode()); assert self.ctx
    def _s(self,p):
        if not p: raise RuntimeError(self.L.parakeet_capi_last_error(self.ctx).decode())
        s=ctypes.string_at(p).decode(); self.L.parakeet_capi_free_string(p); return s
    def vad(self,pcm,opts=None):
        pcm=np.ascontiguousarray(pcm,dtype=np.float32)
        o=json.dumps(opts).encode() if opts else None
        return json.loads(self._s(self.L.parakeet_capi_vad_pcm_json(self.ctx,pcm.ctypes.data,len(pcm),16000,o)))
    def words_batch(self,clips):
        cat=np.ascontiguousarray(np.concatenate(clips),dtype=np.float32)
        n=np.array([len(c) for c in clips],dtype=np.int32)
        return json.loads(self._s(self.L.parakeet_capi_transcribe_pcm_batch_json(self.ctx,cat.ctypes.data,n.ctypes.data,len(clips),16000,2)))

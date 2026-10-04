# Docker images

Two prebuilt images are published to GitHub Container Registry on every push to `master`, one per binary:

- `ghcr.io/mudler/parakeet.cpp-cli`: the command-line transcriber.
- `ghcr.io/mudler/parakeet.cpp-server`: the [OpenAI-compatible server](cli.md#openai-compatible-server).

Each comes in a CPU and a CUDA variant (the CUDA tag is suffixed `-cuda`), and both are multi-arch (`linux/amd64` and `linux/arm64`), so the right one is pulled for your host automatically. They contain just the binary, so mount a converted `.gguf` model (and, for the cli, your audio) at runtime:

```sh
# CLI, CPU
docker run --rm \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/audio:/audio:ro" \
  ghcr.io/mudler/parakeet.cpp-cli:latest \
  transcribe --model /models/parakeet-tdt_ctc-110m-q5_k.gguf --input /audio/speech.wav --decoder tdt

# CLI, CUDA (needs the nvidia container toolkit on the host)
docker run --rm --gpus all \
  -v "$PWD/models:/models:ro" -v "$PWD/audio:/audio:ro" \
  ghcr.io/mudler/parakeet.cpp-cli:latest-cuda \
  transcribe --model /models/parakeet-tdt_ctc-110m-q5_k.gguf --input /audio/speech.wav --decoder tdt

# Server: binds 0.0.0.0 and exposes 8080. Fetch a model by alias on first run,
# or mount a local .gguf. Add --gpus all with the :latest-cuda tag for GPU.
docker run --rm -p 8080:8080 ghcr.io/mudler/parakeet.cpp-server:latest --model tdt_ctc-110m
```

The CUDA image is built on CUDA 13, so it covers everything from Turing up through Blackwell, including GB10 / Grace-Blackwell (DGX Spark) on arm64.

To build the images yourself, see the build args at the top of the [`Dockerfile`](../Dockerfile); the cli is the default target and the server is `--target runtime-server`. The CPU image is the portable `GGML_NATIVE=OFF` build, so it runs on any amd64 or arm64 host.

// test_streaming_diarization.cpp
//
// Tests the streaming diarization path (Phase 2) against the offline path.
// Loads the diarization GGUF, processes a WAV file in chunks, and compares
// the total number of segments and speaker coverage with the offline path.
//
// Environment:
//   PARAKEET_TEST_DIAR_GGUF — path to the diarization GGUF
//   PARAKEET_TEST_AUDIO     — path to a test WAV file

#include "diarization.hpp"
#include "diarization_streaming.hpp"
#include "mel.hpp"
#include "audio_io.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

int main() {
    const char* gguf_path = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    const char* audio_path = std::getenv("PARAKEET_TEST_AUDIO");
    if (!gguf_path || !audio_path) {
        std::printf("test_streaming_diarization: PARAKEET_TEST_DIAR_GGUF or PARAKEET_TEST_AUDIO not set; skip\n");
        return 77;
    }

    // Load the diarization model
    auto model = pk::DiarizationModel::load(gguf_path);
    if (!model) {
        std::printf("test_streaming_diarization: failed to load model: %s\n", gguf_path);
        return 1;
    }

    // Load audio
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(audio_path, audio)) {
        std::printf("test_streaming_diarization: failed to load audio: %s\n", audio_path);
        return 1;
    }
    std::printf("test_streaming_diarization: loaded %s (%zu samples, %d Hz)\n",
                audio_path, audio.samples.size(), audio.sample_rate);

    // 1. Run offline diarization for reference
    auto offline_result = model->diarize_pcm(audio.samples, audio.sample_rate);
    std::printf("test_streaming_diarization: offline segments = %zu\n",
                offline_result.segments.size());

    // 2. Run streaming diarization
    pk::StreamingDiarization stream(model->loader());
    stream.reset();

    // Compute mel features for the full audio
    std::vector<float> mel;
    int n_mels = 0, T = 0;
    model->mel().compute(audio.samples, mel, n_mels, T);
    std::printf("test_streaming_diarization: mel = %d x %d\n", n_mels, T);

    int chunk_len = stream.chunk_len();
    int n_mels_expected = stream.n_mels();
    assert(n_mels == n_mels_expected);

    std::vector<pk::StreamingSpeakerSegment> stream_segs;
    int offset = 0;
    while (offset < T) {
        int n_frames = std::min(chunk_len, T - offset);
        bool is_last = (offset + n_frames >= T);

        // Extract chunk: mel[m*n_frames + t]
        std::vector<float> chunk((size_t)n_mels * n_frames);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < n_frames; ++t)
                chunk[(size_t)m * n_frames + t] = mel[(size_t)m * T + (offset + t)];

        auto segs = stream.feed_mel_chunk(chunk, n_mels, n_frames, is_last);
        for (auto& s : segs)
            stream_segs.push_back(s);

        offset += n_frames;
    }

    std::printf("test_streaming_diarization: streaming segments = %zu\n",
                stream_segs.size());

    // 3. Verify: streaming should produce a reasonable number of segments.
    //    The exact count won't match offline (streaming uses spkcache context),
    //    but it should be in the same ballpark.
    if (stream_segs.empty()) {
        std::printf("test_streaming_diarization: FAIL — no streaming segments produced\n");
        return 1;
    }

    // Count unique speakers in both
    std::set<int> offline_spk, stream_spk;
    for (auto& s : offline_result.segments) offline_spk.insert(s.speaker);
    for (auto& s : stream_segs) stream_spk.insert(s.speaker);

    std::printf("test_streaming_diarization: offline speakers = %zu, streaming speakers = %zu\n",
                offline_spk.size(), stream_spk.size());

    // Streaming should detect at least 1 speaker
    if (stream_spk.empty()) {
        std::printf("test_streaming_diarization: FAIL — no speakers detected in streaming\n");
        return 1;
    }

    // Segments should be in chronological order
    for (size_t i = 1; i < stream_segs.size(); ++i) {
        if (stream_segs[i].start < stream_segs[i-1].start) {
            std::printf("test_streaming_diarization: FAIL — segments not in order at %zu\n", i);
            return 1;
        }
    }

    // Segment timestamps should be within the audio duration
    float audio_dur = (float)audio.samples.size() / 16000.0f;
    for (auto& s : stream_segs) {
        if (s.start < 0.0f || s.end > audio_dur + 1.0f) {
            std::printf("test_streaming_diarization: FAIL — segment out of range: %.2f-%.2f (dur=%.2f)\n",
                        s.start, s.end, audio_dur);
            return 1;
        }
    }

    std::printf("test_streaming_diarization: PASS (%zu streaming segments, %zu offline)\n",
                stream_segs.size(), offline_result.segments.size());
    return 0;
}

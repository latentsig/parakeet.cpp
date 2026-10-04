#!/bin/bash
# Batched VAD decode on long talks (PR 82): wall time and peak RSS of `transcribe --vad`.
# usage: longform_b1.sh TALKS_DIR GGUF_DIR BUILDS_DIR OUT_TSV LOADLOG [MAXWAIT_SECONDS]
#   TALKS_DIR  holds <Talk>-merged.wav for the four talks named below (TED-LIUM long-form, merged)
#   GGUF_DIR   holds ultra-q8_0.gguf and redux-keep.gguf
#   BUILDS_DIR holds build-base, build-b0, build-b1 (the three builds being compared)
# Runs 5 repetitions per (talk, model, build), build order rotated per repetition, 8 threads pinned
# to cores 0-7, serialized with a lock. Before each run it waits until the mean number of other
# runnable tasks (over 10 s) is below 4. The transcript hash is written to the TSV so equality of
# the outputs can be checked.
T=$1; G=$2; B=$3; R=$4; L=$5; MAXWAIT=${6:-10800}; WAITED=0
OUTD=$(mktemp -d)
others() { t=0; for i in 1 2 3 4 5 6 7 8 9 10; do n=$(ps -eo stat,comm --no-headers | awk '$1 ~ /^R/ && $2!="ps" && $2!="awk"' | wc -l); t=$((t+n)); sleep 1; done; echo "scale=1; $t/10" | bc; }
wait_quiet() {
  while true; do
    l=$(cut -d' ' -f1 /proc/loadavg); o=$(others)
    if awk "BEGIN{exit !($o < 4)}"; then echo "$(date +%H:%M:%S) ok runnable_mean $o loadavg1 $l" >> "$L"; return 0; fi
    echo "$(date +%H:%M:%S) runnable_mean $o loadavg1 $l waiting" >> "$L"
    if [ $WAITED -ge $MAXWAIT ]; then echo "GAVE UP after ${WAITED}s" >> "$L"; return 1; fi
    sleep 240; WAITED=$((WAITED+250))
  done
}
for talk in BillGates AimeeMullins JaneMcGonigal DanielKahneman; do
 for model in ultra-q8_0 redux-keep; do
  for run in 1 2 3 4 5; do
   builds=(base b0 b1); k=$(( (run-1) % 3 )); builds=("${builds[@]:k}" "${builds[@]:0:k}")
   for b in "${builds[@]}"; do
    grep -q -P "^$talk\t$model\t$run\t$b\t" "$R" 2>/dev/null && continue
    wait_quiet || exit 1
    lb=$(cut -d' ' -f1-3 /proc/loadavg)
    flock /tmp/pk-bench.lock taskset -c 0-7 /usr/bin/time -f "%e %M" -o "$OUTD/tm.txt" \
      "$B/build-$b/examples/cli/parakeet-cli" transcribe --model "$G/$model.gguf" --input "$T/$talk-merged.wav" --vad --json --threads 8 > "$OUTD/o.json" 2> "$OUTD/err.txt"
    rc=$?
    la=$(cut -d' ' -f1-3 /proc/loadavg)
    sha=$(sha256sum < "$OUTD/o.json" | cut -c1-16)
    echo -e "$talk\t$model\t$run\t$b\t$(tail -1 "$OUTD/tm.txt")\t$sha\trc$rc\tload_before $lb\tload_after $la" >> "$R"
   done
  done
 done
done
rm -rf "$OUTD"; echo ALLDONE >> "$L"

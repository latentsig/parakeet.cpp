#!/bin/sh
# Regenerates every table in results/ from the decode cache (no ASR run).
# Run it from the work directory (or set PK_WORK); the scripts are next to this file.
# PYTHON is the interpreter (needs numpy, soundfile, jiwer).
HERE="$(cd "$(dirname "$0")" && pwd)"
export PK_WORK="${PK_WORK:-$(pwd)}"
P="${PYTHON:-python3}"; S="$HERE/scripts"; O="$PK_WORK/results"; mkdir -p "$O"
$P $S/report_grid.py redux all T0,T0.1,T0.2,T0.3,T0.5,T1.0 > $O/grid_redux_all.md
for s in tune heldout; do $P $S/report_grid.py redux $s T0,T0.1,T0.2,T0.3,T0.5,T1.0 > $O/grid_redux_$s.md; done
for d in ultra v3; do for s in tune heldout; do $P $S/report_cand.py $d $s T0,T0.3,T0.5,P0.5/0.3,E0.5 T0 white5,pink5,pink0 > $O/cand_${d}_$s.md; done; done
for s in tune heldout; do $P $S/report_cand.py redux $s T0,T0.1,T0.2,T0.3,T0.32,T0.35,T0.5,T1.0,P0.5/0.3,E0.5,E1.0,E0.5p0.5,L3 T0 white5,white0,pink5,pink0 > $O/cand_redux_$s.md; done
for d in ultra v3; do $P $S/table_trim.py T0,T0.1,T0.2,T0.3,T0.5,T1.0 all $d | grep talks > $O/grid_${d}_talks_all.txt; done
for d in redux ultra v3; do for k in talk sinr; do $P $S/boundary.py $d T0 T0.3 all $k; done; done > $O/boundary_T0_T0.3.txt
for d in redux ultra v3; do for p in "T0 T0.3" "T0.3 T0.5"; do $P $S/segchange.py $d $p talk,sinr; done; done > $O/segchange.txt 2>&1
$P $S/segchange.py redux T0.3 T0.32 talk >> $O/segchange.txt; $P $S/segchange.py redux T0.3 T0.35 talk >> $O/segchange.txt
$P $S/clip_count.py T0.1,T0.2,T0.3,T0.5,T1.0,P0.5/0.3 > $O/clip_count.txt 2>&1
$P $S/audio_clip.py T0.1,T0.2,T0.3,T0.4,T0.5,T1.0,P0.5/0.3,P0.5/0.4 ultra,redux,v3 > $O/audio_clip.txt 2>&1
$P $S/vad_edges.py > $O/vad_edges.txt 2>&1
$P $S/vad_cover.py > $O/vad_cover.txt 2>&1
$P $S/removed.py > $O/removed.txt 2>&1
$P $S/noise_ins.py T0,T0.1,T0.3,T0.5,T1.0,P0.5/0.3,E0.5,E1.0,E0.5p0.5,L3,L6 > $O/noise_seconds_90_inserts.txt 2>&1
$P $S/noise_words.py T0,T0.3,T0.5,P0.5/0.3,E0.5,L3 redux,ultra,v3 > $O/noise_words_20_inserts.md 2>&1
$P $S/pooled.py > $O/pooled.md

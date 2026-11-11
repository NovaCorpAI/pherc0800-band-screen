#!/bin/bash
# pod_acw.sh : after pod_lane.sh is DONE with the CW winner, resume the ACW short fit to the full horizon too, snap its
# MAXW windings, run seam_test.py on both final fitted sets and pack results_acw.tar. Same conventions as pod_lane.sh.
set -uo pipefail
W=${W:-/workspace/w}; PKG=$W/lane0800; SP=$PKG/spiral_old; SPY=$W/envs/spiral/bin/python
LANE=${LANE:-$W/lane}; LANE_ENV=${LANE_ENV:-$PKG/lane.env}; source $LANE_ENV
MAXW=${MAXW:-40}; WCENTER=${WCENTER:-}; STEPS=${STEPS:-30000}; S=$SCROLL; DS=$W/ds
CORES=$(nproc); TH=${THREADS:-$CORES}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PATH=$W/envs/spiral/bin:$W/bin:$HOME/.local/bin:$PATH
SNAP_PROCS=$(( CORES > 8 ? 8 : CORES ))
log() { echo "$(date +%s) $*" >> $LANE/logs/acw.log; }
[ -f $LANE/DONE ] || { log "lane not DONE yet"; exit 1; }
tag=${TB}acw; short=$(ls -d $LANE/fits/*_${TB}acws | head -1)
run_of() { ls -d $LANE/fits/*_$1 2>/dev/null | head -1; }
if [ ! -f "$(run_of $tag)/checkpoint_fitted.ckpt" ]; then
  log "fit $tag start (resume from $short)"
  ( cd $SP && env FIT_SPIRAL_RESUME_PATH=$short/checkpoint_fitted.ckpt FIT_SPIRAL_OUT_DIR=$LANE/fits FIT_SPIRAL_RUN_TAG=$tag PYTHONPATH=$PKG/native_pkg \
      FIT_SPIRAL_CONFIG_OVERRIDES="{\"z_begin\": $Z0, \"z_end\": $Z1, \"input_disable_patches\": true, \"loss_weight_shell_outer\": 0, \"loss_weight_shell_patch_radius\": 0, \"dense_spacing_mode\": \"phase\", \"optimizer_num_training_steps\": $STEPS}" \
      $SPY fit_spiral.py --dataset $DS/${S}_ACW --cache $LANE/cache ) > $LANE/logs/fit_$tag.log 2>&1
  log "fit $tag end rc=$?"
fi
[ -f "$(run_of $tag)/checkpoint_fitted.ckpt" ] || { log "FAILED no checkpoint"; exit 1; }
run=$(run_of $tag); src=$run/meshes/fitted_$tag; o=$LANE/out/$tag; mkdir -p $o/fitted $o/snap $o/logs
for d in $src/w[0-9][0-9][0-9]_$tag; do [ -f $d/x.tif ] && ln -sfn $d $o/fitted/$(basename $d _$tag); done
ls $o/fitted | grep -E '^w[0-9]{3}$' | sort > $o/windings_all.txt
$SPY -c "
ws=[l.strip() for l in open('$o/windings_all.txt') if l.strip()]; n=sorted(int(w[1:]) for w in ws); m=int('$WCENTER') if '$WCENTER' else n[len(n)//2]
open('$o/windings.txt','w').write('\n'.join(sorted(ws, key=lambda w: (abs(int(w[1:])-m), w))[:$MAXW])+'\n')"
if [ ! -f $o/snap/.done ]; then
  log "snap $tag start"
  SURF_ZARR=$DS/m7_$S $SPY $PKG/snap_meshes.py $o/fitted $S $o/snap --windings $o/windings.txt --z0 $Z0 --z1 $Z1 --level 0 --procs $SNAP_PROCS > $o/logs/snap.log 2>&1 && touch $o/snap/.done
  log "snap $tag end rc=$?"
fi
best=$(cat $LANE/best_tag)
for t in $best $tag; do $SPY $PKG/seam_test.py $DS/m7_$S $LANE/out/$t/fitted $LANE/out/$t/windings.txt --label final_$t 2>/dev/null; done > $LANE/seam_final.json
for t in $best $tag; do awk -F'\t' -v t=$t 'NR>1{s+=$4; r+=$7; o+=$5; n++} END {printf "%s n=%d snapped=%.4f rejected=%.4f med_off=%.3f\n", t, n, s/n, r/n, o/n}' $LANE/out/$t/snap/snap_report.tsv; done > $LANE/snap_final.txt
for t in $best $tag; do echo "== $t"; tr '\r' '\n' < $LANE/logs/fit_$t.log | grep -E '^step [0-9]+: loss|satisfied_tracks' | tail -n 2; done > $LANE/loss_final.txt
R=$LANE/results_acw; rm -rf $R; mkdir -p $R/snap $R/fitted $R/logs
cp $o/windings.txt $o/windings_all.txt $o/snap/snap_report.tsv $LANE/seam_final.json $LANE/snap_final.txt $LANE/loss_final.txt $R/ 2>/dev/null
for w in $(cat $o/windings.txt); do
  [ -f $o/snap/$w/x.tif ] && cp -r $o/snap/$w $R/snap/$w
  [ -f $o/fitted/$w/x.tif ] && { mkdir -p $R/fitted/$w; cp -L $o/fitted/$w/*.tif $o/fitted/$w/*.json $R/fitted/$w/ 2>/dev/null; }
done
tail -c 30000 $LANE/logs/fit_$tag.log > $R/logs/fit_$tag.log; cp $o/logs/snap.log $R/logs/ 2>/dev/null; cp $LANE/logs/acw.log $R/logs/
tar -C $LANE -cf $LANE/results_acw.tar results_acw && log "DONE_ACW $(du -sh $LANE/results_acw.tar | cut -f1)"

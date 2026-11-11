#!/bin/bash
# pod_lane.sh : ONE band of one scroll on a RunPod box, up to the snapped meshes (flatten, render, ink and triage run on Kaggle).
# 306B's x_vast_plan/lane_box.sh (MIT) cut down to the stages that need the 24 GB GPU and the dataset next to the bucket:
#   00 disk    free space on $W >= NEED_GB
#   01 setup   pod_setup.sh (envs, patches, native modules, VC3D for the axis)
#   02 umb     build_umb.py (xy normal grids every 50 slices over [Z0-600, Z1+600] -> vc_gen_umbilicus) + smooth_umb.py (sigma 75)
#   03 data    pod_data.sh (tracks, lasagna g2, m7 L0+L1, CT L4+L1, surf-SDT, resident pools) + the ACW twin dataset
#   04 fit     SENSE=AB: both senses to SHORT_N steps (FIT_SPIRAL_STOP_AT), snap test of each (sense_geo.py, else fewer
#              rejections wins), the winner resumes from its checkpoint to STEPS. SENSE=CW|ACW: one full fit.
#   05 snap    snap_meshes.py (343) on the MAXW windings nearest the median winding (middle-out)
#   06 pack    $LANE/results.tar: snapped + fitted tifxyz of those windings, snap reports, sense, timing, logs
# Idempotent and resumable: every stage leaves $LANE/stages/<stage>.ok; a failure writes $LANE/FAILED and exits 1.
# Config (LANE_ENV, shell syntax): SCROLL Z0 Z1 TB UM SENSE [MAXW=40] [WCENTER=<median>] [SHORT_N=4000] [STEPS=30000] [NEED_GB=70]
set -uo pipefail
W=${W:-/workspace/w}; PKG=$W/lane0800; SP=$PKG/spiral_old; SPY=$W/envs/spiral/bin/python
LANE=${LANE:-$W/lane}; LANE_ENV=${LANE_ENV:-$PKG/lane.env}; source $LANE_ENV
: "${SCROLL:?}" "${Z0:?}" "${Z1:?}" "${TB:?}" "${UM:?}" "${SENSE:?}"
MAXW=${MAXW:-40}; WCENTER=${WCENTER:-}; SHORT_N=${SHORT_N:-4000}; NEED_GB=${NEED_GB:-70}; STEPS=${STEPS:-30000}
S=$SCROLL; DS=$W/ds; UMB=$LANE/umb_${S}_s75.json
mkdir -p $LANE/stages $LANE/logs $LANE/results $DS
op=$(cat $LANE/pid 2>/dev/null); if [ -n "$op" ] && [ "$op" != "$$" ] && kill -0 $op 2>/dev/null; then echo "pod_lane already running (pid $op)"; exit 0; fi
echo $$ > $LANE/pid
CORES=$(nproc); TH=${THREADS:-$CORES}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMBA_NUM_THREADS=$TH PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PATH=$W/envs/spiral/bin:$W/bin:$HOME/.local/bin:$PATH AWS_CONFIG_FILE=$PKG/aws_fast.cfg
MEMB=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); [[ "$MEMB" =~ ^[0-9]+$ ]] || MEMB=$(awk '/MemTotal/{print $2*1024}' /proc/meminfo)
MEM_GB=$(( MEMB / 1073741824 )); SDTW=$(( CORES > 4 ? CORES - 2 : 2 )); SDTM=$(( MEM_GB * 10 / 25 )); [ $SDTM -lt 2 ] && SDTM=2
[ $SDTW -gt $SDTM ] && SDTW=$SDTM   # make_surf_sdt: ~2.5 GB per worker
export W PKG DSROOT=$DS SPIRAL_DIR=$SP SPIRAL_PY=$SPY SDT_WORKERS=$SDTW SNAP_PROCS=$(( CORES > 8 ? 8 : CORES ))
CUR=init
echo "$(date +%s) start $S $Z0-$Z1 sense $SENSE tag $TB cores $CORES mem ${MEM_GB}G sdt_workers $SDTW" >> $LANE/logs/lane.log
( while true; do echo "$(date +%s) $(cat $LANE/cur_stage 2>/dev/null)" > $LANE/heartbeat; sleep 60; done ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT

fail() { echo "$CUR: $*" > $LANE/FAILED; echo "$(date +%s) FAILED $CUR: $*" >> $LANE/logs/lane.log; exit 1; }
stage() {   # stage NAME FUNC : run FUNC once, time it
  CUR=$1; [ -f $LANE/stages/$1.ok ] && return 0
  rm -f $LANE/FAILED; echo "$1" > $LANE/cur_stage; local t0=$(date +%s)
  "$2" || fail "stage function returned $?"
  local t1=$(date +%s); printf '%s\t%s\t%s\t%s\n' "$1" $t0 $t1 $((t1 - t0)) >> $LANE/timing.tsv; touch $LANE/stages/$1.ok
  echo "$(date +%s) ok $1 in $((t1 - t0)) s" >> $LANE/logs/lane.log
}

s_disk() {
  local free=$(df --output=avail -BG $W | tail -1 | tr -dc 0-9)
  echo "free ${free} GB, need $NEED_GB GB" >> $LANE/logs/lane.log
  [ "$free" -ge "$NEED_GB" ] || { echo "only ${free} GB free, need $NEED_GB" > $LANE/logs/disk.txt; return 1; }
}
s_setup() { bash $PKG/pod_setup.sh > $LANE/logs/setup.log 2>&1; grep -q SETUP_OK $LANE/logs/setup.log; }
s_umb() {
  $SPY $PKG/build_umb.py $S $((Z0 - 600)) $((Z1 + 600)) 50 $LANE/umb_${S}_raw.json > $LANE/logs/umb.log 2>&1 || return 1
  $SPY $PKG/smooth_umb.py $LANE/umb_${S}_raw.json $UMB --sigma 75 --step 25 >> $LANE/logs/umb.log 2>&1 || return 1
  $SPY -c "import json;c=json.load(open('$UMB'))['control_points'];print('umb points',len(c),'z',c[0]['z'],c[-1]['z'])" >> $LANE/logs/umb.log
}
s_data() {
  bash $PKG/pod_data.sh $S $Z0 $Z1 CW $UMB > $LANE/logs/data.log 2>&1 || return 1
  grep -q DATA_OK $LANE/logs/data.log || return 1
  local A=$DS/${S}_ACW; mkdir -p $A                              # ACW twin: shares everything but the scroll spec
  for f in tracks lasagna_inputs umbilicus.json; do ln -sfn $DS/$S/$f $A/$f; done
  $SPY -c "import json;d=json.load(open('$DS/$S/spiral-scroll.json'));d['spiral_outward_sense']='ACW';json.dump(d,open('$A/spiral-scroll.json','w'))"
}
runfit() {   # runfit DSNAME TAG [extra env assignments...]
  local dsn=$1 tag=$2; shift 2
  [ -n "$(ls -d $LANE/fits/*_$tag 2>/dev/null)" ] && [ -f "$(ls -d $LANE/fits/*_$tag | head -1)/checkpoint_fitted.ckpt" ] && return 0
  echo "$(date +%s) fit $tag start ($*)" >> $LANE/logs/lane.log
  ( cd $SP && env "$@" FIT_SPIRAL_OUT_DIR=$LANE/fits FIT_SPIRAL_RUN_TAG=$tag PYTHONPATH=$PKG/native_pkg${PYTHONPATH:+:$PYTHONPATH} \
      FIT_SPIRAL_CONFIG_OVERRIDES="{\"z_begin\": $Z0, \"z_end\": $Z1, \"input_disable_patches\": true, \"loss_weight_shell_outer\": 0, \"loss_weight_shell_patch_radius\": 0, \"dense_spacing_mode\": \"phase\", \"optimizer_num_training_steps\": $STEPS}" \
      $SPY fit_spiral.py --dataset $DS/$dsn --cache $LANE/cache ) > $LANE/logs/fit_$tag.log 2>&1
  local rc=$?; echo "$(date +%s) fit $tag end rc=$rc" >> $LANE/logs/lane.log; return $rc
}
run_of() { ls -d $LANE/fits/*_$1 2>/dev/null | head -1; }
link_meshes() {   # link_meshes TAG -> $LANE/out/TAG/fitted/wNNN
  local tag=$1 run=$(run_of $1); local src=$run/meshes/fitted_$tag; local o=$LANE/out/$tag; mkdir -p $o/fitted $o/snap $o/logs
  for d in $src/w[0-9][0-9][0-9]_$tag; do [ -f $d/x.tif ] && ln -sfn $d $o/fitted/$(basename $d _$tag); done
  ls $o/fitted | grep -E '^w[0-9]{3}$' | sort > $o/windings_all.txt
  [ -s $o/windings_all.txt ] || { echo "no fitted meshes under $src" >> $LANE/logs/lane.log; return 1; }
}
middle_out() {   # middle_out TAG N -> $LANE/out/TAG/windings.txt: the N windings nearest WCENTER (default: the median winding)
  local o=$LANE/out/$1; $SPY -c "
ws=[l.strip() for l in open('$o/windings_all.txt') if l.strip()]; n=sorted(int(w[1:]) for w in ws); m=int('$WCENTER') if '$WCENTER' else n[len(n)//2]
open('$o/windings.txt','w').write('\n'.join(sorted(ws, key=lambda w: (abs(int(w[1:])-m), w))[:$2])+'\n')"
}
snap_tag() {   # snap_tag TAG : link the fit's meshes, snap the middle-out MAXW windings (the sense test reads snap/snap_report.tsv)
  local tag=$1 o=$LANE/out/$1; link_meshes $tag || return 1; middle_out $tag $MAXW; [ -f $o/snap/.done ] && return 0
  SURF_ZARR=$DS/m7_$S $SPY $PKG/snap_meshes.py $o/fitted $S $o/snap \
    --windings $o/windings.txt --z0 $Z0 --z1 $Z1 --level 0 --procs $SNAP_PROCS > $o/logs/snap.log 2>&1 && touch $o/snap/.done
}
s_fit() {
  # drop corrupt .npz caches (a track-crossing cache cut off by a killed run) so the fitter rebuilds them instead of dying
  find $DS -name '*.npz' -size +0 2>/dev/null | while read f; do $SPY -c "import sys,zipfile; sys.exit(0 if zipfile.is_zipfile('$f') else 1)" || { echo "corrupt cache removed: $f" >> $LANE/logs/lane.log; rm -f "$f"; }; done
  case $SENSE in
    CW)  runfit $S ${TB}cw || return 1; echo ${TB}cw > $LANE/best_tag ;;
    ACW) runfit ${S}_ACW ${TB}acw || return 1; echo ${TB}acw > $LANE/best_tag ;;
    AB)
      runfit $S ${TB}cws FIT_SPIRAL_STOP_AT=$SHORT_N || return 1
      runfit ${S}_ACW ${TB}acws FIT_SPIRAL_STOP_AT=$SHORT_N || return 1
      snap_tag ${TB}cws || return 1; snap_tag ${TB}acws || return 1
      local g=$($SPY $PKG/sense_geo.py $LANE/out/${TB}cws $LANE/out/${TB}acws 2>> $LANE/sense.txt)
      if [ -z "$g" ]; then   # not decisive: lower mean sheet-jump rejection wins, flagged weak
        g=$($SPY -c "
import csv
m=lambda d: sum(float(r['frac_rejected']) for r in csv.DictReader(open(d+'/snap/snap_report.tsv'),delimiter='\t'))
print('ACW' if m('$LANE/out/${TB}acws') < m('$LANE/out/${TB}cws') else 'CW')"); echo "weak: tie-break by mean rejections -> $g" >> $LANE/sense.txt
      fi
      echo "sense=$g (short fits at $SHORT_N steps)" >> $LANE/sense.txt
      local s=$(echo $g | tr A-Z a-z); local dsn=$S; [ $g = ACW ] && dsn=${S}_ACW
      runfit $dsn ${TB}$s FIT_SPIRAL_RESUME_PATH=$(run_of ${TB}${s}s)/checkpoint_fitted.ckpt || return 1
      echo ${TB}$s > $LANE/best_tag ;;
    *) return 1 ;;
  esac
  [ -f "$(run_of $(cat $LANE/best_tag))/checkpoint_fitted.ckpt" ]
}
s_snap() { snap_tag $(cat $LANE/best_tag); }
s_pack() {
  local tag=$(cat $LANE/best_tag) R=$LANE/results o; o=$LANE/out/$tag
  rm -rf $R; mkdir -p $R/snap $R/fitted $R/logs
  cp $LANE/timing.tsv $LANE/sense.txt $LANE/best_tag $LANE_ENV $UMB $LANE/umb_${S}_raw.json $R/ 2>/dev/null
  cp $o/windings.txt $o/windings_all.txt $o/snap/snap_report.tsv $R/ 2>/dev/null
  cp $DS/$S/spiral-scroll.json $R/; cp $DS/$S/umbilicus.json $R/umbilicus_used.json
  for w in $(cat $o/windings.txt); do
    [ -f $o/snap/$w/x.tif ] && cp -r $o/snap/$w $R/snap/$w
    [ -f $o/fitted/$w/x.tif ] && { mkdir -p $R/fitted/$w; cp -L $o/fitted/$w/*.tif $o/fitted/$w/*.json $R/fitted/$w/ 2>/dev/null; }
  done
  for t in ${TB}cws ${TB}acws; do [ -f $LANE/out/$t/snap/snap_report.tsv ] && cp $LANE/out/$t/snap/snap_report.tsv $R/snap_report_$t.tsv; done
  local run=$(run_of $tag); ls $run > $R/fit_run_listing.txt 2>/dev/null; cp $run/*.json $R/ 2>/dev/null
  for f in lane.log umb.log setup.log; do tail -c 200000 $LANE/logs/$f > $R/logs/$f 2>/dev/null; done
  tail -c 50000 $LANE/logs/data.log > $R/logs/data.log 2>/dev/null
  for f in $LANE/logs/fit_*.log; do [ -f $f ] && tail -c 30000 $f > $R/logs/$(basename $f); done
  cp $o/logs/snap.log $R/logs/ 2>/dev/null
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $R/gpu.txt 2>/dev/null; nproc > $R/cores.txt
  tar -C $LANE -cf $LANE/results.tar results && du -sh $LANE/results.tar >> $LANE/logs/lane.log
}

stage 00_disk s_disk
stage 01_setup s_setup
stage 02_umb s_umb
stage 03_data s_data
stage 04_fit s_fit
stage 05_snap s_snap
stage 06_pack s_pack
echo "$(date +%s) DONE" >> $LANE/logs/lane.log; date +%s > $LANE/DONE

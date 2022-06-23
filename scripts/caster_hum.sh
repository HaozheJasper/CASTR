#!/usr/bin/env bash
date=`date "+%Y-%m-%d-%H%M%S"`
nohup python launch_experiment.py "configs/humanoid-dir.json"  --gpu $1 --rew_fac "25,100" --gnn "221_64,128" --memo "caster r25 m-net def" >> $date.out &
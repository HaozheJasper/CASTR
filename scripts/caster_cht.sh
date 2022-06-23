#!/usr/bin/env bash
date=`date "+%Y-%m-%d-%H%M%S"`
nohup python launch_experiment.py "configs/cheetah-vel.json"  --gpu $1 --rew_fac "10,100" --gnn "221_64,32" --memo "caster r10 m-net def" >> $date.out &
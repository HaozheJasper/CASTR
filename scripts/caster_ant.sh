#!/usr/bin/env bash
date=`date "+%Y-%m-%d-%H%M%S"`
nohup python launch_experiment.py "configs/ant-goal.json"  --gpu $1 --rew_fac "10,100" --gnn "221_64,64" --memo "caster r10 l-net def" >> $date.out &
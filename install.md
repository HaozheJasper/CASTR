# mujoco
[site](https://www.roboti.us/index.html)
1. download zip [win](https://www.roboti.us/download/mjpro150_win64.zip),[linux](https://www.roboti.us/download/mjpro150_linux.zip).
2. place the key file to `.mujoco/mjkey.txt`
3. unzip to wherever, but need to set `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MJPRO_PATH`.
# pip
pip install -U 'mujoco-py<1.50.2,>=1.50.1'
pip install gtimer gym tb-nightly click
# conda
conda install -c fastai fastai
conda install -c pytorch pytorch
conda install joblib pandas
conda install -c conda-forge python-dateutil

# rand_params_envs
obtain from [here](https://github.com/dennisl88/rand_param_envs).

unzip and `pip install -e .`
pip install gym==0.7.4 mujoco-py==0.5.7
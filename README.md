This repository is based on [NeRF-- repository](https://github.com/ActiveVisionLab/nerfmm)

Dataset required is the modified LLFF dataset used in NeRF--

## Environment

```sh
conda env create -f environment.yml
```

## Training

```sh
python run_nerf.py --base_dir='.' --scene_name='LLFF/fern'
```

## Viewing the learning process

```sh
tensorboard --logdir=logs/mnerf/LLFF/fern/ --port=6006
```

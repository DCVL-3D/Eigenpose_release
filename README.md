# Eigenpose_release
This repository is an official Pytorch implementation of the paper "Eigenpose: Occlusion-robust 3D Human Mesh Reconstruction".

## Installation
### Environment Setting
Please set the environment and datasets by following the guidance of [ROMP](https://github.com/Arthur151/ROMP).

### Model Checkpoints (only for evaluation)
Create `checkpoints` directory and place the [model checkpoint](https://drive.google.com/file/d/1mCYygCFms7fT-9VWDiLwisW1xnEN-XEw/view?usp=sharing) file in it.

## Run
### Evaluation
```
$ python -m romp.test --configs_yml=configs/eval_3dpw_test_resnet.yml
$ python -m romp.test --configs_yml=configs/eval_ochuman_resnet_test.yml
$ python -m romp.test --configs_yml=configs/eval_crowdpose_test.yml
$ python -m romp.test --configs_yml=configs/eval_oh50k_test.yml
```
※ You can change the subset of 3DPW (3DPW-PC, 3DPW-OC) by changing the `eval_dataset` setting in the file `configs/eval_3dpw_test_resnet.yml`.

## Results
#### Results of 3D human mesh reconstruction by the proposed method on 3DPW (1st row) and COCO (2nd and 3rd rows) datasets.
![Figure8](./figure/fig8.svg)

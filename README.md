# Object level Visual Reasoning in Videos


This repository contains a Pytorch implementation of ["Object level Visual Reasoning in Videos"](https://arxiv.org/abs/1806.06157), [F. Baradel](https://fabienbaradel.github.io/), [N. Neverova](https://nneverova.github.io/), [C. Wolf](https://perso.liris.cnrs.fr/christian.wolf/), [J. Mille](http://www.rfai.li.univ-tours.fr/PagesPerso/jmille/), [G. Mori](http://www.cs.sfu.ca/~mori/), In ECCV 2018.

Links: [Project page](https://fabienbaradel.github.io/eccv18_object_level_visual_reasoning/) | [Camera-ready](https://fabienbaradel.github.io/papers/ECCV_18) | [Complementary Mask Data](https://fabienbaradel.github.io/masks_data/)

<img src="img/teaser_carrots.png" width="800"/>

## TODO
* meta files:
  * `manifest.txt`: a list of all video folders; may not be needed if 
  * `splitId.txt`: correpsonding to `manifest.txt`; e.g. split ids for VLOG: test:0 / val:3 / train:1+2

## Code
We release code for training and testing our implementation.
We encourage you to follow the steps below:
* [preprocessing the video dataset](./preprocessing/README.md)
    * rescaling an entire dataset (WxH=256x256 and fps=30)
* [testing the dataloader](./loader/README.md)
    * efficient video decoding on the fly
* [training/testing the model](./README_TRAINING_TESTING.md)
    * training procedure using precomputed masks

## Masks
Please visit the following [website](https://fabienbaradel.github.io/masks_data/) for downloading the mask predictions.

# Requirements
* pytorch 0.4.0
* numpy
* [lintel](https://github.com/dukebw/lintel) - make sure that you have already installed this library (important for decoding videos on the fly)

## Citation
If you find this paper or our implementation useful for your research or if you use the precomputed masks, please cite our paper.
```
@InProceedings{Baradel_2018_ECCV,
author = {Baradel, Fabien and Neverova, Natalia and Wolf, Christian and Mille, Julien and Mori, Greg},
title = {Object Level Visual Reasoning in Videos},
booktitle = {ECCV},
year = {2018}
}
```

## Acknowledgements
This work was funded by grant Deepvision (ANR-15- CE23-0029, STPGP-479356-15), a joint French/Canadian call by ANR & NSERC.

## Licence
MIT License

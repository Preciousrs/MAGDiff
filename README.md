# MAGDiff: A Synergistic Multi-Attribute Guided Diffusion Framework for Personalized Fashion Garment Generation
## Overview
 <img src="https://github.com/Preciousrs/MTTV/blob/main/fakeddit_s.png" width="800" height="300" /> 
 
## Installation
1.Clone the repository
```
git clone https://github.com/Preciousrs/MAGDiff.git
cd MAGDiff
```
2.Install Python dependencies
```
conda create -n MAGDiff python=3.10
pip install -r requirements.txt
conda activate MAGDiff
```
3.Download the pretrained `Realistic_Vision_V4.0_noVAE`, SD 1.5 `image_encoder` checkpoint and put it in `models/`.

##Training
Run:
```
sh train.sh
```

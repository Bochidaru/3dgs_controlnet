0. clone repo:
- git clone https://github.com/Bochidaru/3dgs_controlnet.git
- cd 3dgs_controlnet

1. install env:
- conda env update --file environment.yaml --prune (for lightning ai)

2. get the dataset:
- gdown 1SUl2oOsD_rvNr2wu1KY16aGRibYCGBmZ
- unzip

3. get the sd1.5 weight from hf:
- curl -L -o v1-5-pruned.ckpt https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt

4. run this for loading weight from sd1.5:
- python tool_add_control.py v1-5-pruned.ckpt .\models\control_sd15_ini.ckpt 

5. run tutorial_train.py 

HAVE FUN!!
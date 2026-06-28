0. clone repo:
- git clone https://github.com/Bochidaru/3dgs_controlnet.git
- cd 3dgs_controlnet

1. install env:
- conda env update --file environment.yaml --prune         (for lightning ai)

2. get the dataset:
- option 1:  gdown 1SUl2oOsD_rvNr2wu1KY16aGRibYCGBmZ
- option 2:  hf download minhphuong04/3dgs_cldm_dataset_and_weight cldm_dataset.zip --repo-type=dataset --local-dir .         (fastest)
- unzip

3. get the sd1.5 weight from hf:
- option 1: gdown 1fcikb_mcSGsIhh4XrntiLTFaEPmcdmta
- option 2: hf download stable-diffusion-v1-5/stable-diffusion-v1-5 v1-5-pruned.ckpt --repo-type=model --local-dir .
- option 3: hf download minhphuong04/3dgs_cldm_dataset_and_weight v1-5-pruned.ckpt --repo-type=dataset --local-dir .

4. run this for loading weight from sd1.5:
- python tool_add_control.py v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt 

5. run tutorial_train.py :
- python tutorial_train.py



For continue training from checkpoint: 
- Skip step 3 and 4, download newest weight from hf:  https://huggingface.co/datasets/minhphuong04/3dgs_cldm_dataset_and_weight/tree/main
- hf download minhphuong04/3dgs_cldm_dataset_and_weight LATEST_WEIGHT_HERE --repo-type=dataset --local-dir ./models/
- Example: hf download minhphuong04/3dgs_cldm_dataset_and_weight weights-epoch=30-step=2000.ckpt --repo-type=dataset --local-dir ./models/
- in tutorial_train, fill path for new ckpt file in this line `resume_ckpt_path = ""`
- python tutorial_train.py


For upload ckpt to hf repo:
- hf auth login
- hf upload minhphuong04/3dgs_cldm_dataset_and_weight ./weights/{fill here} --repo-type=dataset 

HAVE FUN!!
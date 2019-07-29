# MobileDentist

/checkpoints/ is the place to put trained models, and we now have 3 versions for the oral case: VGG_SSD, VGG_SSD with fine tuning, and UNET_SSD. Please help check their saved models on [google drive](https://drive.google.com/drive/folders/1ZbDpv40x1kVBbN3-54egi8C15MtebLlY?usp=sharing).

/engines/ is the place for entries of inference/training. For simple inference with one given image, please check the script engine_simple_infer_SSD.py. Others engine files are for training/batch evaluating purposes.

/demo/ is the place for storing images for simple style inference. For the oral case, I've put an example image example.jpg from testing group in that folder. All other files now are for debugging purposes. 

The project is written in pytorch, and has a dependency on [mmcv](https://github.com/open-mmlab/mmcv), which is a basic CV library. 

For running the inference on server, follow the steps:
1. install requirements
2. change $LD_LIBRARY_PATH to current env bin path.
3. change $PYTHONPATH to the home folder of the project /path/to/MobileDentist. 

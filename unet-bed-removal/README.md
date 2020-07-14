
# Unet: remove bed from CT images

This project aims to automatically remove bed/ pillow/ cover from original CT images. 

The model only supports Grayscale CT.

You can download the well trained model from https://drive.google.com/file/d/1uwYvPBrlLnjPrEqk8qcAZxv7ov60ETds/view?usp=sharing

## Usage
**Note : Use Python 3**
### Bed removal

You can use output.py to remove the bed. Follow these steps:

1. Change directory of target image. If you need assess the algorithm, you have to put annotated CTs in 'mask' folder.

`python output.py -asm`

2. Choose a model. This project also provides a training part based on Unet which you can make model suit your data.

`python output.py -asm -m model.pth`

```shell script
> python output.py - h
usage: output.py  [-m] [--model FILE] 
                  [-asm] [--assessment]
optional arguments:
  -h, --help            show this help message and exit
  --asm ASM, -a ASM     Assess recovered images compared with ground truth
                        (default: False)
  --model FILE, -m FILE
                        Specify the model file (default: model.pth)
````
The input images and annotated images should be in the `data/prediction/img` and `data/prediction/mask` folders.

The output mask (non-blurred) and recovered images should be in the `data/prediction/pred_mask` and `data/predition/mapping` folders.

### Prediction

You can easily test the output masks on your images via predict.py

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```
You can specify which model file to use with `--model MODEL.pth`.

### Training

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 10.0)
```
By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

The input images and target masks should be in the `data/training/img` and `data/training/full_mask` folders respectively.

The training results (accuracy, f1-score, dice coefficient, loss) are in `results` folder.

A suggestion: The black holes of target mask should be filled. You can use `floodfill.py` to fill the holes. It's better to use .bmp format instead of other compressed format when use floodfill algorithm.

---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)

Referenced github project milesial/Pytorch-UNet. [https://github.com/milesial/Pytorch-UNet]



import cv2
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from unet import UNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
import time
import logging
from dice_loss import dice_coeff
import argparse

# in_path1 for test img, in_path2 for GT of test img
# out_path for predicted full_mask, out_path2 for recovered images
in_path = 'D:/unet-bed-removal/data/prediction/img/'
in_path2 = 'D:/unet-bed-removal/data/prediction/mask/'
out_path = 'D:/unet-bed-removal/data/prediction/pred_mask/'
out_path2 = 'D:/unet-bed-removal/data/prediction/mapping/'

model = 'model.pth'

def prediction(net, imgs, device):
    net.eval()
    ds = BasicDataset('data/training/img', 'data/training/full_mask', scale=0.5)
    img = ds.preprocess(imgs)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(imgs.size[1]),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
    return full_mask > 0.5
def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))
def image_to_mask(image):
    image[image < 128] = 0
    image[image >= 128] = 1
    return image
def blurred(in_im, val1, val2):
    # greater val1 and val2 have a better blur effect
    out_im = np.hstack([cv2.blur(in_im, (val1, val2))])
    return out_im
def binary(data):
    data[data <= 0.5] = 0
    data[data > 0.5] = 1
    return data
def assessment(path1, path2):
    ds = BasicDataset(path1, path2, scale=0.5)
    loader = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True)
    tp, tn, fn, fp, dice, count = [0, 0, 0, 0, 0, 0]
    for batch in loader:
        count += 1
        true_mask = batch['image']
        pred_mask = batch['mask']
        pred_mask = pred_mask.to(device='cpu' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        true_mask = true_mask.to(device='cpu' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        for GT, pred in zip(true_mask, pred_mask):
            pred = binary(pred).float()
            GT = binary(GT).float()
            tp += ((pred == 1) & (GT == 1)).cpu().sum().numpy()
            tn += ((pred == 0) & (GT == 0)).cpu().sum().numpy()
            fn += ((pred == 0) & (GT == 1)).cpu().sum().numpy()
            fp += ((pred == 1) & (GT == 0)).cpu().sum().numpy()
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * r * p / (r + p)
            acc = (tp + tn) / (tp + tn + fp + fn)
            dice += dice_coeff(pred, GT.squeeze(dim=1)).item()
    return p, r, f1, acc, dice/count

def get_args():
    parser = argparse.ArgumentParser(description='Bed removal and assessment',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--asm', '-a',
                        help="Assess recovered images compared with ground truth",
                        default=False)
    parser.add_argument('--model', '-m', default='model.pth',
                        metavar='FILE',
                        help="Specify the model file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    count = 0
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    start = time.time()
    net = UNet(n_channels=1, n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    if args.model:
        net.load_state_dict(torch.load(args.model, map_location=device))
    else:
        net.load_state_dict(torch.load(model, map_location=device))
    logging.info(f'''Starting bed removal:
        Device:          {device.type}
        Model:           {model}
    ''')
    for fn in os.listdir(in_path):
        count += 1
        img = Image.open(in_path+fn)
        mask = prediction(net=net, imgs=img, device=device)
        result=mask_to_image(mask)
        result.save(out_path+fn)
        result2 = image_to_mask(np.array(result))
        blur_img = blurred(result2, 20, 20)
        blurred_img = mask_to_image(blur_img)
        blurred_img.save(in_path2+fn)
        mapping = img*blur_img
        Image.fromarray(mapping).save(out_path2+fn)

    if args.asm:
        p, r, f1, acc, dice = assessment(in_path2, out_path2)
        logging.info("Assessing recovered images ...")
        logging.info(f'''Assessment Outcome:
             mean precision:          {p}
             mean recall:             {r}
             mean accuracy:           {acc}
             mean f1-score:           {f1}
             mean dice-coefficient    {dice}
            ''')
    end = time.time()
    logging.info('Number of images: {}'.format(count))
    logging.info('run time: {}'.format(end-start))
    logging.info('run time per image: {}'.format((end-start)/count))



import torch
from tqdm import tqdm


from dice_loss import dice_coeff


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    dice = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tot = 0
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            img = batch['image']
            true_mask = batch['mask']

            imgs = img.to(device=device, dtype=torch.float32)
            true_masks = true_mask.to(device=device, dtype=torch.float32)
            mask_pred = net(imgs)

            for true_mask, pred in zip(true_masks, mask_pred):

                pred = (pred>0.5).float()
                tp += ((pred == 1) & (true_mask == 1)).cpu().sum().numpy()
                tn += ((pred == 0) & (true_mask == 0)).cpu().sum().numpy()
                fn += ((pred == 0) & (true_mask == 1)).cpu().sum().numpy()
                fp += ((pred == 1) & (true_mask == 0)).cpu().sum().numpy()

                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * r * p / (r + p)
                acc = (tp + tn) / (tp + tn + fp + fn)

                dice += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])

    return dice / n_val, acc, f1, img, true_mask, mask_pred

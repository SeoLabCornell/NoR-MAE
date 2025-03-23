import argparse
import numpy as np
import os
import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import util.misc as misc
import matplotlib.pyplot as plt
import seaborn as sns

from timm.data.loader import MultiEpochsDataLoader
from timm.models.vision_transformer import PatchEmbed

img_size = 224
patch_size = 16
in_channels=3
embed_dim=192

# neighbor 
k_num = 1

random_state = 170

# patch embedding
patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
common_params = {
    "n_init": "auto",
    "random_state": random_state,
}


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=2048, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Dataset parameters
    parser.add_argument('--data_path', default='/share/seo/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    parser.add_argument('--multi_epochs_dataloader', action='store_true', help='Use MultiEpochsDataLoader to prevent reinitializing dataloader per epoch')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary masks
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # neighbor
    sim_matrix = sim_patches(x)

    print(sim_matrix.mean())
    print(sim_matrix.min())
    print(sim_matrix.max())

    plt.figure(figsize=(16, 8))
    sns.histplot(sim_matrix.view(-1).detach().cpu().numpy(), stat="probability")
    plt.xlim(left=0.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.savefig("./similarity_plot/similarity_batch.png", dpi=300)
    plt.close()

    return x_masked, mask, ids_restore, sim_matrix

def get_local_index(N_patches, k_size):
    """
    Get the local neighborhood of each patch 
    """
    loc_weight = []
    w = torch.LongTensor(list(range(int(math.sqrt(N_patches)))))
    for i in range(N_patches):
        ix, iy = i//len(w), i%len(w)
        wx = torch.zeros(int(math.sqrt(N_patches)))
        wy = torch.zeros(int(math.sqrt(N_patches)))
        wx[ix]=1
        wy[iy]=1
        for j in range(1,int(k_size//2)+1):
            wx[(ix+j)%len(wx)]=1
            wx[(ix-j)%len(wx)]=1
            wy[(iy+j)%len(wy)]=1
            wy[(iy-j)%len(wy)]=1
        weight = (wy.unsqueeze(0)*wx.unsqueeze(1)).view(-1)
        weight[i] = 0
        loc_weight.append(weight.nonzero().squeeze())
    return torch.stack(loc_weight)

def sim_patches(x):
    N, L, D = x.shape
    loc224 = get_local_index(196, 3)
    loc224 = loc224.to(x.device)
    
    x_norm = nn.functional.normalize(x, dim=-1)
    sim_matrix = x_norm[:,loc224] @ x_norm.unsqueeze(2).transpose(-2,-1)
    top_idx = sim_matrix.squeeze().topk(k=k_num,dim=-1)[1].view(N, L, k_num, 1)
    top_sim = sim_matrix.squeeze().topk(k=k_num,dim=-1)[0].view(N, L, k_num, 1)

    return top_sim

def patchify(imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_embed.patch_size[0]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_embed.patch_size[0]
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def run_one_image(img, mask_ratio=0.74):
    patch_embed.to(img.device)
    xs = patch_embed(img)
    xs_masked, mask, ids_restore, sim_matrix = random_masking(xs, mask_ratio)


def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    args.input_size = 224

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    dataloader_cls = MultiEpochsDataLoader if args.multi_epochs_dataloader else torch.utils.data.DataLoader

    data_loader_train = dataloader_cls(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    for idx, (samples, _) in enumerate(data_loader_train):
        samples = samples.to(device, non_blocking=True)
        print(samples.shape)
        break

    run_one_image(samples, 0.0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

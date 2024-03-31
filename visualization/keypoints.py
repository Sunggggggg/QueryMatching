import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from data import download

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

benchmark = 'spair'
datapath = '/home/dev4/data/SKY/datasets'
thres = 'box'
device = torch.device('cuda')

val_dataset = download.load_dataset(benchmark, datapath, thres, device, 'val')
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=True)

mini_batch = val_dataloader[0]

# #of keypoint
n_pts = mini_batch['n_pts']

# Source image
src_img = mini_batch['src_img']
src_kps = mini_batch['src_kps']

src_img_np = src_img[0].permute(1, 2, 0).numpy()    # [h, w, 3]
src_img_np = to8b(src_img_np)

src_kps = src_kps[0, :,  :n_pts].to(torch.int32)    # [2, 9]
for i in range(n_pts):
    plt.scatter(src_kps[0, i], src_kps[1, i])

plt.imshow(src_img_np)
plt.axis('off')
plt.show()

# Target image
tar_img = mini_batch['trg_img']
tar_kps = mini_batch['trg_kps']
n_pts = mini_batch['n_pts']

tar_img_np = tar_img[0].permute(1, 2, 0).numpy()    # [h, w, 3]
tar_img_np = to8b(tar_img_np)

tar_kps = tar_kps[0, :,  :n_pts].to(torch.int32)    # [2, 9]
for i in range(n_pts):
    plt.scatter(tar_kps[0, i], tar_kps[1, i])

plt.imshow(src_img_np)
plt.axis('off')
plt.show()
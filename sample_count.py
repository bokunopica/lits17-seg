import os
import os
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as tt
import numpy as np
from tqdm import tqdm


def fname_to_ids(fname):
    """
    return ct_id and ct_slice_id
    volume-001-058.png -> 1, 58
    """
    seg_list = fname.split('-')
    return seg_list[1], seg_list[2].replace('.png', '')


if __name__ == "__main__":
    data_dir = "/home/qianq/data/lits17_png"
    source_list = os.listdir(os.path.join(data_dir, 'source'))
    SINGLE_TOTAL = 512*512
    mode = torchvision.io.image.ImageReadMode.RGB
    class_cnt = [0, 0, 0]
    for i in tqdm(range(len(source_list))):
        fname = source_list[i]
        ct_id, slice_id = fname_to_ids(fname)
        ct_id = int(ct_id)
        slice_id = int(slice_id)
        target_fname_liver = "seg-liver-%03d-%03d.png" % (ct_id, slice_id)
        target_fname_tumor = "seg-tumor-%03d-%03d.png" % (ct_id, slice_id)
        # feature = torchvision.io.read_image(
        #     os.path.join(data_dir, 'source', fname),
        #     mode,
        # )
        label_liver = torchvision.io.read_image(os.path.join(data_dir, 'target', target_fname_liver), mode)
        label_tumor = torchvision.io.read_image(os.path.join(data_dir, 'target', target_fname_tumor), mode)
        total = SINGLE_TOTAL
        li_cnt = int(label_liver.count_nonzero()/3)
        tu_cnt = int(label_tumor.count_nonzero()/3)
        if li_cnt:
            total -= li_cnt
        if tu_cnt:
            total -= tu_cnt
        class_cnt[0] += total
        class_cnt[1] += li_cnt
        class_cnt[2] += tu_cnt
    print(class_cnt) # [15001262016, 332214453, 18462603]
    # 45:1 812:1         
    # 近似于~810:18:1
        

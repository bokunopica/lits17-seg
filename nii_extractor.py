import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import trange

# nifti file load
def read_nii_file(nii_path):
    '''
    根据路径读取文件
    '''
    nii_img=nib.load(nii_path)
    # 可以获取很多信息包括shape，仿射值，数据类型和矩阵
    # print(nii_img)
    # load的包的数据类型 <class 'nibabel.nifti1.Nifti1Image'>
    # print(type(nii_img))
    # print(nii_img.shape)
    # (611, 512, 512)
    return nii_img


def nii_one_slice(image_arr, index):
    '''
    显示nii image中的其中一张slice
    '''
    # 查看图像的长宽高 发现和nifti包的shape是一样的(611, 512, 512)
    # 注意：nibabel读出的image的data的数组顺序为：Width，Height，Channel
    # print(image_arr.shape)
    # 将2d数组转置，就是调换xy的位置
    # image_2d = image_arr[0,:,:].transpose((1, 0))
    # 当然也可以不调换
    image_2d = image_arr[:, :, index]
    plot_img(image_2d)


def plot_img(image, title=""):
    plt.imshow(image, cmap='gray', )
    plt.axis('off')
    plt.title(title)
    plt.show()


def get_nii_one_slice(image_arr, index):
    '''
    获取nii image中的其中一张slice
    '''
    # image_2d = image_arr[0,:,:].transpose((1, 0))
    image_2d = image_arr[:, :, index].transpose((1,0))
    return image_2d


def split_target_img(image):
    img1, img2 = [], []
    for x in np.nditer(image):
        if x == 1.:
            img1.append(255.)
            img2.append(0.)
        elif x== 2.:
            img1.append(0.)
            img2.append(255.)
        else:
            img1.append(0.)
            img2.append(0.)
    img1 = np.array(img1).reshape(image.shape)
    img2 = np.array(img2).reshape(image.shape)
    return img1, img2



def window_transform(ct_array, window_width, window_center, normal=False):
   """
   return: trucated image according to window center and window width
   and normalized to [0,1]
   """
   minWindow = float(window_center) - 0.5*float(window_width)
   newimg = (ct_array - minWindow) / float(window_width)
   newimg[newimg < 0] = 0
   newimg[newimg > 1] = 1
   if not normal:
        newimg = (newimg * 255).astype('uint8')
   return newimg

def window_transform_single(ct_img, window_width, window_center, normal=False):
   """
   return: trucated image according to window center and window width
   and normalized to [0,1]
   """
   minWindow = float(window_center) - 0.5*float(window_width)
   newimg = (ct_img - minWindow) / float(window_width)
   newimg[newimg < 0] = 0
   newimg[newimg > 1] = 1
   if not normal:
        newimg = (newimg * 255).astype('uint8')
   return newimg

if __name__ == "__main__":
    input_dir = "/home/qianq/data/lits17_raw"
    output_dir = "/home/qianq/data/lits17_png"
    scan_dir = f"{input_dir}/scan"
    label_dir = f"{input_dir}/label"
    window_width = 200
    window_center = 50
    for i in trange(1, 131):
        source_path = f"{scan_dir}/volume-{i}.nii"
        target_path = f"{label_dir}/segmentation-{i}.nii"
        source = read_nii_file(source_path)
        target = read_nii_file(target_path)
        source_img_arr = source.get_fdata()
        target_img_arr = target.get_fdata()
        slice_cnt = source.shape[2]
        # for _index in range(slice_cnt):
        for _index in trange(slice_cnt):
            single_source = get_nii_one_slice(source_img_arr, _index)
            single_target = get_nii_one_slice(target_img_arr, _index)
            # target中 data = 1.0 肝 data = 2.0 肿瘤 需要分别处理保存
            liver_target, tumor_target = split_target_img(single_target)
            single_source_img = Image.fromarray(
                window_transform_single(
                    single_source,
                    window_width=window_width,
                    window_center=window_center,
                )
            ).convert('L')
            liver_target_img = Image.fromarray(liver_target).convert('1')
            tumor_target_img = Image.fromarray(tumor_target).convert('1')
            single_source_img.save(f"{output_dir}/source/volume-{'%03d'%i}-{'%03d'%_index}.png")
            liver_target_img.save(f"{output_dir}/target/seg-liver-{'%03d'%i}-{'%03d'%_index}.png")
            tumor_target_img.save(f"{output_dir}/target/seg-tumor-{'%03d'%i}-{'%03d'%_index}.png")
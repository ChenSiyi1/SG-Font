import os
import torch
import numpy as np
import tqdm
import argparse
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS
from pytorch_fid import fid_score
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-gt', '--gt_dir', type=str, default='path/to/data')
parser.add_argument('-pred', '--result_dir', type=str, default='path/to/data')
parser.add_argument('-o', '--out', type=str, default='metrics.txt')
parser = parser.parse_args()

result_dir = parser.result_dir
ground_truth_dir = parser.gt_dir
output_file = parser.out

lpips_net = LPIPS(net="alex").cuda()

# 获取所有子目录的名称
indexes = os.listdir(result_dir)
indexes = sorted(indexes,  key=lambda k: int(k.lstrip('id_')))
fs_bar = tqdm.tqdm(indexes)

l1_values = []
rmse_values = []
ssim_values = []
lpips_values = []
# fid_values = []

l1_mean = 0
rmse_mean = 0
ssim_mean = 0
lpips_mean = 0
# fid_mean = 0

def main():
    for ind in fs_bar:
        result_path = os.path.join(result_dir, ind)
        gt_path = os.path.join(ground_truth_dir, ind)

        # 获取该类别下所有图片的文件名
        filenames = sorted(os.listdir(result_path))

        l1_sum = 0
        rmse_sum = 0
        ssim_sum = 0
        lpips_sum = 0
        fid_sum = 0

        for filename in filenames:
            # 读取结果图像和对应的ground truth图像
            result_img = lpips.load_image(os.path.join(result_path, filename)) # HWC, RGB, [0, 255]
            gt_img = lpips.load_image(os.path.join(gt_path, filename))

            # 计算L1
            result_array = np.array(result_img).astype(np.float32) / 255.0 # HWC, [0,1]
            gt_array = np.array(gt_img).astype(np.float32) / 255.0
            l1_sum += np.mean(np.abs(result_array - gt_array))            # Smaller, better

            # 计算RMSE
            rmse_sum += np.sqrt(np.mean((result_array - gt_array) ** 2))

            # 计算SSIM
            # ssim_sum += ssim(result_array, gt_array, multichannel=True)
            ssim_sum += cal_MSSIM(result_img, gt_img)

            # 计算LPIPS
            lpips_value = cal_lpips(result_img, gt_img)
            lpips_sum += lpips_value

            # 计算FID
            # result_tensor = (torch.from_numpy(result_array.transpose(2, 0, 1)).unsqueeze(0).cuda() - 0.5) * 2
            # gt_tensor = (torch.from_numpy(gt_array.transpose(2, 0, 1)).unsqueeze(0).cuda() - 0.5) * 2
            # fid_value = fid_score(result_tensor, gt_tensor, cuda=True)
            # fid_sum += fid_value.cpu().detach().numpy()

        num_images = len(filenames)
        l1_values.append(l1_sum / num_images)
        rmse_values.append(rmse_sum / num_images)
        ssim_values.append(ssim_sum / num_images)
        lpips_values.append(lpips_sum / num_images)
        # fid_values.append(fid_sum / num_images)

    # 计算所有类别的均值
    l1_mean = np.mean(l1_values)
    rmse_mean = np.mean(rmse_values)
    ssim_mean = np.mean(ssim_values)
    lpips_mean = np.mean(torch.tensor(lpips_values).cpu().detach().numpy())
    # fid_mean = np.mean(fid_values)

    # 将结果保存到txt文件中
    with open(output_file, "a") as f:
        for i, id in enumerate(indexes):
            f.write("Id: {}\n".format(id))
            f.write("L1: {}\n".format(l1_values[i]))
            f.write("RMSE: {}\n".format(rmse_values[i]))
            f.write("SSIM: {}\n".format(ssim_values[i]))
            f.write("LPIPS: {}\n".format(lpips_values[i]))
            # f.write("FID: {}\n".format(fid_values[i]))
            f.write("\n")
        f.write("Mean L1: {}\n".format(l1_mean))
        f.write("Mean RMSE: {}\n".format(rmse_mean))
        f.write("Mean SSIM: {}\n".format(ssim_mean))
        f.write("Mean LPIPS: {}\n".format(lpips_mean))
        # f.write("Mean FID: {}\n".format(fid_mean))


def cal_MSSIM(imgs_fake, imgs_real):
        mssim0 = ssim(imgs_fake[:, :, 0], imgs_real[:, :, 0], data_range=255, gaussian_weights=True)
        mssim1 = ssim(imgs_fake[:, :, 1], imgs_real[:, :, 1], data_range=255, gaussian_weights=True)
        mssim2 = ssim(imgs_fake[:, :, 2], imgs_real[:, :, 2], data_range=255, gaussian_weights=True)
        mssim = (mssim0 + mssim1 + mssim2) / 3
        return mssim


def cal_lpips(i0, i1):
    img0 = lpips.im2tensor(i0)  # [-1, 1]
    img1 = lpips.im2tensor(i1)

    img0 = img0.cuda()
    img1 = img1.cuda()
    # Compute distance
    dist01 = lpips_net.forward(img0, img1).flatten()  # RGB image from [-1,1]
    assert len(dist01) == 1
    return dist01[0]


if __name__ == "__main__":
    main()


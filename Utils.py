import numpy as np
from scipy.fftpack import dct, idct
import random
import os
import torch
from operator import truediv


# DCT
def dctn(x, norm="ortho"):
    for i in range(x.ndim):
        x = dct(x, axis=i, norm=norm)
    return x


# IDCT
def idctn(x, norm="ortho"):
    for i in range(x.ndim):
        x = idct(x, axis=i, norm=norm)
    return np.around(np.maximum(np.real(x), 0), decimals=2)


# Gauss high-pass filter
def high_pass_filter(shape, sigma):
    x, y = np.ogrid[-shape[0] // 2:shape[0] // 2, -shape[1] // 2:shape[1] // 2]
    return 1 - np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


# Gauss low-pass filter
def low_pass_filter(shape, sigma):
    x, y = np.ogrid[-shape[0] // 2:shape[0] // 2, -shape[1] // 2:shape[1] // 2]
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


# LiDAR Gauss high-pass filter
def lidar_high_pass(image, sigma):
    filter_kernel = high_pass_filter(image.shape, sigma)
    result = image * filter_kernel
    return result


# LiDAR Gauss low-pass filter
def lidar_low_pass(image, sigma):
    # 创建高通滤波器
    filter_kernel = low_pass_filter(image.shape, sigma)
    # 滤波
    result = image * filter_kernel
    return result


# HSI Gauss high-pass filter
def hsi_high_pass(image, sigma):
    result = np.zeros(image.shape, dtype=complex)
    filter_kernel = high_pass_filter(image.shape, sigma)

    for band in range(image.shape[-1]):
        band_data = image[..., band]
        filtered_band = band_data * filter_kernel
        result[..., band] = filtered_band

    return result


# HSI Gauss low-pass filter
def hsi_low_pass(image, sigma):
    result = np.zeros(image.shape, dtype=complex)
    filter_kernel = low_pass_filter(image.shape, sigma)

    for band in range(image.shape[-1]):
        band_data = image[..., band]
        filtered_band = band_data * filter_kernel
        result[..., band] = filtered_band

    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def generate_mask(sample_num, cls_num, data_label):
    #SEED =
    #set_seed(SEED)

    N = sample_num
    cls = cls_num
    data_gt = data_label
    cou = np.zeros(shape=(20,))
    count = 0
    gt_train = np.zeros((data_gt.shape[0], data_gt.shape[1]))
    while count < (N*cls):
        x = random.randint(0, data_gt.shape[0] - 1)
        y = random.randint(0, data_gt.shape[1] - 1)
        if data_gt[(x, y)] == 1 and gt_train[(x, y)] == 0 and cou[1] < N:
            gt_train[(x, y)] = 1
            cou[1] = cou[1] + 1
            count = count + 1

        if data_gt[(x, y)] == 2 and gt_train[(x, y)] == 0 and cou[2] < N:
            gt_train[(x, y)] = 1
            cou[2] = cou[2] + 1
            count = count + 1

        if data_gt[(x, y)] == 3 and gt_train[(x, y)] == 0 and cou[3] < N:
            gt_train[(x, y)] = 1
            cou[3] = cou[3] + 1
            count = count + 1

        if data_gt[(x, y)] == 4 and gt_train[(x, y)] == 0 and cou[4] < N:
            gt_train[(x, y)] = 1
            cou[4] = cou[4] + 1
            count = count + 1

        if data_gt[(x, y)] == 5 and gt_train[(x, y)] == 0 and cou[5] < N:
            gt_train[(x, y)] = 1
            cou[5] = cou[5] + 1
            count = count + 1

        if data_gt[(x, y)] == 6 and gt_train[(x, y)] == 0 and cou[6] < N:
            gt_train[(x, y)] = 1
            cou[6] = cou[6] + 1
            count = count + 1

        if data_gt[(x, y)] == 7 and gt_train[(x, y)] == 0 and cou[7] < N:
            gt_train[(x, y)] = 1
            cou[7] = cou[7] + 1
            count = count + 1

        if data_gt[(x, y)] == 8 and gt_train[(x, y)] == 0 and cou[8] < N:
            gt_train[(x, y)] = 1
            cou[8] = cou[8] + 1
            count = count + 1

        if data_gt[(x, y)] == 9 and gt_train[(x, y)] == 0 and cou[9] < N:
            gt_train[(x, y)] = 1
            cou[9] = cou[9] + 1
            count = count + 1

        if data_gt[(x, y)] == 10 and gt_train[(x, y)] == 0 and cou[10] < N:
            gt_train[(x, y)] = 1
            cou[10] = cou[10] + 1
            count = count + 1
        if data_gt[(x, y)] == 11 and gt_train[(x, y)] == 0 and cou[11] < N:
            gt_train[(x, y)] = 1
            cou[11] = cou[11] + 1
            count = count + 1

        if data_gt[(x, y)] == 12 and gt_train[(x, y)] == 0 and cou[12] < N:
            gt_train[(x, y)] = 1
            cou[12] = cou[12] + 1
            count = count + 1

        if data_gt[(x, y)] == 13 and gt_train[(x, y)] == 0 and cou[13] < N:
            gt_train[(x, y)] = 1
            cou[13] = cou[13] + 1
            count = count + 1

        if data_gt[(x, y)] == 14 and gt_train[(x, y)] == 0 and cou[14] < N:
            gt_train[(x, y)] = 1
            cou[14] = cou[14] + 1
            count = count + 1

        if data_gt[(x, y)] == 15 and gt_train[(x, y)] == 0 and cou[15] < N:
            gt_train[(x, y)] = 1
            cou[15] = cou[15] + 1
            count = count + 1

        if data_gt[(x, y)] == 16 and gt_train[(x, y)] == 0 and cou[16] < N:
            gt_train[(x, y)] = 1
            cou[16] = cou[16] + 1
            count = count + 1

        if data_gt[(x, y)] == 17 and gt_train[(x, y)] == 0 and cou[17] < N:
            gt_train[(x, y)] = 1
            cou[17] = cou[17] + 1
            count = count + 1

        if data_gt[(x, y)] == 18 and gt_train[(x, y)] == 0 and cou[18] < N:
            gt_train[(x, y)] = 1
            cou[18] = cou[18] + 1
            count = count + 1

        if data_gt[(x, y)] == 19 and gt_train[(x, y)] == 0 and cou[19] < N:
            gt_train[(x, y)] = 1
            cou[19] = cou[19] + 1
            count = count + 1
    gt_train = data_gt * gt_train
    gt_train = gt_train.astype(int)
    return gt_train, SEED


def sample_wise_normalization(data):
    _min = data.min()
    _max = data.max()
    return (data - _min) / (_max - _min)+0.001


def split_data(data_hsi, data_lidar, label, r):
    patch_hsi = []
    patch_lidar = []
    patch_label = []
    result_hsi = []
    result_lidar = []
    result_label = []
    for i in range(r, label.shape[0]-r):
        for j in range(r, label.shape[1]-r):
            if(label[i][j]>0):
                patch_hsi.append(data_hsi[i-r:i+r+1, j-r:j+r+1, ...])
                patch_lidar.append(data_lidar[i-r:i+r+1, j-r:j+r+1, ...])
                patch_label.append(label[i, j]-1)
    result_hsi = np.asarray(patch_hsi, dtype=np.float32)
    result_lidar = np.asarray(patch_lidar, dtype=np.float32)
    result_label = np.asarray(patch_label, dtype=np.float32)
    return result_hsi, result_lidar, result_label


def split_data_ex(data_hsi, data_lidar, label, r, hsi_ex, lidar_ex):
    patch_hsi = []
    patch_lidar = []
    patch_hsi_ex = []
    patch_lidar_ex = []
    patch_label = []
    result_hsi = []
    result_lidar = []
    result_hsi_ex = []
    result_lidar_ex = []
    result_label = []
    for i in range(r, label.shape[0]-r):
        for j in range(r, label.shape[1]-r):
            if(label[i][j]>0):
                patch_hsi.append(data_hsi[i-r:i+r+1, j-r:j+r+1, ...])
                patch_lidar.append(data_lidar[i-r:i+r+1, j-r:j+r+1, ...])
                patch_hsi_ex.append(hsi_ex[i-r:i+r+1, j-r:j+r+1, ...])
                patch_lidar_ex.append(lidar_ex[i-r:i+r+1, j-r:j+r+1, ...])
                patch_label.append(label[i, j]-1)
    result_hsi = np.asarray(patch_hsi, dtype=np.float32)
    result_lidar = np.asarray(patch_lidar, dtype=np.float32)
    result_hsi_ex = np.asarray(patch_hsi_ex, dtype=np.float32)
    result_lidar_ex = np.asarray(patch_lidar_ex, dtype=np.float32)
    result_label = np.asarray(patch_label, dtype=np.float32)
    return result_hsi, result_lidar, result_label, result_hsi_ex, result_lidar_ex
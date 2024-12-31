import scipy.io as scio
import numpy as np
import time
from Utils import dctn, idctn, hsi_high_pass, lidar_high_pass, generate_mask, sample_wise_normalization, split_data_ex, split_data


def data_input(dataset, cls):
    print("Data inputing:" + dataset)
    start_time = time.time()

    #Replace the path
    data_path = "data_path"
    data_Lidar_path = "data_Lidar_path"
    datagt_path = "datagt_path"

    data_hsi = scio.loadmat(data_path)
    data_lidar = scio.loadmat(data_Lidar_path)
    data_gt = scio.loadmat(datagt_path)

    data_hsi = data_hsi['data']
    data_lidar = data_lidar['data']
    data_gt = data_gt['label']

    gt_train, SEED = generate_mask(sample_num=20, cls_num=cls, data_label=data_gt)
    scio.savemat(dataset+'_' + str(cls) + '_gt.mat', {'label': gt_train})
    scio.savemat(dataset + '_HSI.mat', {'data': data_hsi})
    scio.savemat(dataset + '_LiDAR.mat', {'data': data_lidar})

    print("End data input:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "耗时%.3fs" % (time.time() - start_time))

    return data_hsi, data_lidar, data_gt, gt_train, SEED


def test_data(dataset, r):
    print("Test data inputing:" + dataset)
    start_time = time.time()

    # Replace the path
    data_path = "data_path"
    data_Lidar_path = "data_Lidar_path"
    datagt_path = "datagt_path"

    data_hsi = scio.loadmat(data_path)
    data_lidar = scio.loadmat(data_Lidar_path)
    data_gt = scio.loadmat(datagt_path)

    data_hsi = data_hsi['data']
    data_lidar = data_lidar['data']
    data_gt = data_gt['label']

    scio.savemat(dataset + '_HSI.mat', {'data': data_hsi})
    scio.savemat(dataset + '_LiDAR.mat', {'data': data_lidar})
    hsi = sample_wise_normalization(data_hsi)
    lidar = sample_wise_normalization(data_lidar)

    hsi_pad = np.lib.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    lidar_pad = np.lib.pad(lidar, ((r, r), (r, r)), 'symmetric')
    gt_pad = np.lib.pad(data_gt, r, 'constant', constant_values=0)

    data_hsi, data_lidar, data_gt = split_data(hsi_pad, lidar_pad, gt_pad, r)

    print("End test data input:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "耗时%.3fs" % (time.time() - start_time))

    return data_hsi, data_lidar, data_gt


def domain_expansion(dataset, cls):
    h, l, gt, gt_train, SEED = data_input(dataset, cls)

    print("HSI expanding...")
    start_time = time.time()

    # Transform HSI to frequency domain
    dct_h = dctn(h)

    high_pass_sigma = 104
    dct_h_high_pass = hsi_high_pass(dct_h, high_pass_sigma)

    decay_coefficient = np.around(np.random.uniform(0.6, 1, (h.shape[0], h.shape[1])), decimals=2)
    dct_h_high_pass = dct_h_high_pass * decay_coefficient[:, :, np.newaxis]

    low_pass = np.random.randint(0, 300, size=(h.shape[0], h.shape[1]))
    h_low_pass = np.stack([low_pass] * h.shape[2], axis=-1)
    dct_h_low_pass = dctn(h_low_pass)

    h_ex = idctn(dct_h_high_pass + dct_h_low_pass)

    print("End HSI expansion:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "耗时%.3fs" % (time.time() - start_time))

    print("LiDAR expanding...")
    start_time = time.time()

    # Transform LiDAR to frequency domain
    dct_l = dctn(l)

    high_pass_sigma = 80
    dct_l_high_pass = lidar_high_pass(dct_l, high_pass_sigma)
    dct_l_low_pass = dct_l - dct_l_high_pass

    decay_coefficient = np.around(np.random.uniform(0.7, 1, (l.shape[0], l.shape[1])), decimals=2)
    dct_l_high_pass = dct_l_high_pass * decay_coefficient

    l_ex = idctn(dct_l_low_pass + dct_l_high_pass)

    print("End LiDAR expansion:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "耗时%.3fs" % (time.time() - start_time))

    return h, l, gt, h_ex, l_ex, gt_train, SEED


def data_process(S, T, cls):
    h, l, gt, h_ex, l_ex, gt_train, SEED = domain_expansion(S, cls)

    print("Train data processing...")
    start_time = time.time()

    hsi = sample_wise_normalization(h)
    lidar = sample_wise_normalization(l)
    hsi_ex = sample_wise_normalization(h_ex)
    lidar_ex = sample_wise_normalization(l_ex)

    r = 5
    hsi_pad = np.lib.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    lidar_pad = np.lib.pad(lidar, ((r, r), (r, r)), 'symmetric')
    hsi_ex_pad = np.lib.pad(hsi_ex, ((r, r), (r, r), (0, 0)), 'symmetric')
    lidar_ex_pad = np.lib.pad(lidar_ex, ((r, r), (r, r)), 'symmetric')
    gt_train_pad = np.lib.pad(gt_train, r, 'constant', constant_values=0)
    gt_pad = np.lib.pad(gt, r, 'constant', constant_values=0)

    hsi_train, lidar_train, label_train, hsi_ex_train, lidar_ex_train = split_data_ex(hsi_pad, lidar_pad, gt_train_pad, r, hsi_ex_pad, lidar_ex_pad)

    print("End train data process:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "耗时%.3fs" % (time.time() - start_time))

    hsi_test, lidar_test, label_test = test_data(T, r)

    return hsi_train, lidar_train, label_train, hsi_test, lidar_test, label_test, SEED, hsi_ex_train, lidar_ex_train

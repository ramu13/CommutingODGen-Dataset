import os
import random

import numpy as np


def load_all_areas(if_shuffle=True):
    areas = os.listdir("data")
    if if_shuffle:
        random.shuffle(areas)
    return areas

def split_train_valid_test(areas, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    # assert train_ratio + valid_ratio + test_ratio == 1

    train_areas = areas[:int(len(areas)*train_ratio)]
    valid_areas = areas[int(len(areas)*train_ratio):int(len(areas)*(train_ratio+valid_ratio))]
    test_areas = areas[int(len(areas)*(train_ratio+valid_ratio)):]
    return train_areas, valid_areas, test_areas

def construct_train(areas):
    xs = []
    ys = []
    for area in areas:
        demos = np.load(f"data/{area}/demos.npy")
        pois = np.load(f"data/{area}/pois.npy")
        dis = np.load(f"data/{area}/dis.npy")
        
        feat = np.concatenate([demos, pois], axis=1)
        # feat_o, feat_d の作成
        feat_o = feat.reshape([feat.shape[0], 1, feat.shape[1]]).repeat(feat.shape[0], axis=1)
        feat_d = feat.reshape([1, feat.shape[0], feat.shape[1]]).repeat(feat.shape[0], axis=0)
        dis = dis.reshape([dis.shape[0], dis.shape[1], 1])
        
        # x: (n, n, 2*feat_dim + 1)
        x = np.concatenate([feat_o, feat_d, dis], axis=2)
        # ※ reshape で1エリア内の各セルの特徴を平坦化した場合、形状は (n*n, 2*feat_dim+1)
        # ここではエリアごとにそのままリストに格納する
        xs.append(x)
        
        # OD行列（そのままの形状、エリアごとにサイズが異なる）
        od = np.load(f"data/{area}/od.npy")
        ys.append(od)
    
    # 連結せず、各エリアごとにリストで返す
    return xs, ys


def construct_validtest(areas):
    x_areas = []
    y_areas = []
    for area in areas:
        demos = np.load(f"data/{area}/demos.npy")
        pois = np.load(f"data/{area}/pois.npy")
        dis = np.load(f"data/{area}/dis.npy")
        
        feat = np.concatenate([demos, pois], axis=1)
        feat_o = feat.reshape([feat.shape[0], 1, feat.shape[1]]).repeat(feat.shape[0], axis=1)
        feat_d = feat.reshape([1, feat.shape[0], feat.shape[1]]).repeat(feat.shape[0], axis=0)
        dis = dis.reshape([dis.shape[0], dis.shape[1], 1])
        
        x = np.concatenate([feat_o, feat_d, dis], axis=2)
        x_areas.append(x)
        
        od = np.load(f"data/{area}/od.npy")
        y_areas.append(od)
    
    return x_areas, y_areas


def load_data(if_shuffle=True):
    areas = load_all_areas(if_shuffle)
    areas = areas[:20]  # For debugging, limit to 20 areas
    train_areas, valid_areas, test_areas = split_train_valid_test(areas, train_ratio=0.1)
    
    x_train, y_train = construct_train(train_areas)
    x_valid, y_valid = construct_validtest(valid_areas)
    x_test, y_test   = construct_validtest(test_areas)
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test
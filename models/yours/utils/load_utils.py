import os
import random
import numpy as np

def load_all_areas(if_shuffle=True):
    """
    Load all areas from the data directory.
    Only the area names are returned.
    """
    areas = os.listdir("data")
    if if_shuffle:
        random.shuffle(areas)
    return areas


def split_train_valid_test(areas, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    Split the areas into training, validation, and test sets.
    Only the area names are returned.
    """
    assert train_ratio + valid_ratio + test_ratio == 1

    train_areas = areas[:int(len(areas)*train_ratio)]
    valid_areas = areas[int(len(areas)*train_ratio):int(len(areas)*(train_ratio+valid_ratio))]
    test_areas = areas[int(len(areas)*(train_ratio+valid_ratio)):]

    return train_areas, valid_areas, test_areas


if __name__ == "__main__":
    root_dir = "data"
    train_areas, valid_areas, test_areas = split_train_valid_test(load_all_areas())
    print("Train Areas:", train_areas)
    print("Valid Areas:", valid_areas)
    print("Test Areas:", test_areas)
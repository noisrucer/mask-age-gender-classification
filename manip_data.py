import os
import shutil
from pathlib import Path

DATA_DIR = Path('./data/FaceMaskClassification')
with_mask_len = len(os.listdir(DATA_DIR / 'with_mask'))
without_mask_len = len(os.listdir(DATA_DIR / 'without_mask'))
print("Original with_mask/ length: {}".format(with_mask_len))
print("Original without_mask/ length: {}".format(without_mask_len))
test_data_split = 0.1
with_mask_len = int(with_mask_len * test_data_split)
without_mask_len = int(without_mask_len * test_data_split)
print("Test data number extracted from with_mask/: {}".format(with_mask_len))
print("Test data number extracted from without_mask/: {}".format(without_mask_len))

cnt = 0
for imgF in os.listdir(DATA_DIR / 'with_mask')[:with_mask_len]:
    shutil.move(
        DATA_DIR / 'with_mask' / imgF,
        DATA_DIR / 'test' / 'with_mask' / imgF
    )

for imgF in os.listdir(DATA_DIR / 'without_mask')[:without_mask_len]:
    shutil.move(
        DATA_DIR / 'without_mask' / imgF,
        DATA_DIR / 'test' / 'without_mask' / imgF
    )
    
print("train/with_mask/ length: {}".format(len(os.listdir(DATA_DIR / 'with_mask'))))
print("train/without_mask/ length: {}".format(len(os.listdir(DATA_DIR / 'without_mask'))))
print("test/with_mask/ length: {}".format(len(os.listdir(DATA_DIR / 'test' / 'with_mask'))))
print("test/without_mask/ length: {}".format(len(os.listdir(DATA_DIR / 'test' / 'without_mask'))))

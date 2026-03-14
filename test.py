import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# 1️⃣ 你的 fMRI 文件
fmri_path = "/Users/linshangjin/Desktop/dachuang/data/peking_test_output_nifty/fmri/fmri_X_1201251_session_1_run1.nii.gz"

# 2️⃣ 下载 Schaefer 200 atlas
atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=200,
    yeo_networks=7,
    resolution_mm=2
)

atlas_filename = atlas.maps

# 3️⃣ 提取时间序列
masker = NiftiLabelsMasker(
    labels_img=atlas_filename,
    standardize=True,
    detrend=True
)

time_series = masker.fit_transform(fmri_path)

print("Time series shape:", time_series.shape)

# 4️⃣ 计算 200×200
corr = np.corrcoef(time_series.T)

print("FC matrix shape:", corr.shape)

# 5️⃣ Fisher Z
corr_z = np.arctanh(corr)

np.save("subject_1038415_fc.npy", corr_z)

print("Saved successfully.")

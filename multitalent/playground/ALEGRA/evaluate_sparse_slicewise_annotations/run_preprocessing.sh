# manually add ignore label to datasets! Remember to remove it once we are done
export nnUNet_preprocessed='/dkfz/cluster/gpu/data/OE0441/isensee/nnUNet_preprocessed_alegra'
multitalent_extract_fingerprint -d 137 216 980 -np 128 --clean
multitalent_plan_experiment -d 137 216 980 -np 32
python generate_configurations.py
python generate_splits.py
multitalent_preprocess -d 137 216 980 -c 3d_fullres 3d_fullres_sparse_anno_slicewise_003 3d_fullres_sparse_anno_slicewise_005 3d_fullres_sparse_anno_slicewise_010 3d_fullres_sparse_anno_slicewise_030 3d_fullres_sparse_anno_slicewise_050 -np 64 64 64 64 64 64

datasets = (137, 216, 980)
base_configuration = '3d_fullres'
new_configurations = {
    '3d_fullres_sparse_anno_slicewise_003': 'SparseSegSliceRandomOrth3',
    '3d_fullres_sparse_anno_slicewise_005': 'SparseSegSliceRandomOrth5',
    '3d_fullres_sparse_anno_slicewise_010': 'SparseSegSliceRandomOrth10',
    '3d_fullres_sparse_anno_slicewise_030': 'SparseSegSliceRandomOrth30',
    '3d_fullres_sparse_anno_slicewise_050': 'SparseSegSliceRandomOrth50',
}
percent_of_cases_annotated = (3, 5, 10, 30, 50, 100)
results_folder = '/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake_alegra_sliceanno'
nnUNet_preprocessed_alegra = '/dkfz/cluster/gpu/data/OE0441/isensee/nnUNet_preprocessed_alegra'
num_runs = 3 # applies to training on fewer train cases
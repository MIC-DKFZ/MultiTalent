from batchgenerators.utilities.file_and_folder_operations import join, load_json
from multitalent.batch_running.learning_from_sparse_annotations.estimate_annotation_percentages import \
    run_on_all_subfolders
from multitalent.playground.ALEGRA.evaluate_sparse_slicewise_annotations.configuration import datasets, nnUNet_preprocessed_alegra
from multitalent.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

if __name__ == '__main__':
    for d in datasets:
        # doent work out of the box because I forgot the '3d_fullres' in the data identifier
        run_on_all_subfolders(join(nnUNet_preprocessed_alegra, maybe_convert_to_dataset_name(d)), n_processes=64)

    # percent annotated based on actual splits (rounding and stuff, especially on 980
    for d in datasets:
        pp_fld = join(nnUNet_preprocessed_alegra, maybe_convert_to_dataset_name(d))
        splits = load_json(join(pp_fld, 'splits_final.json'))
        num_train_cases = len(splits[-1]['train']) # last split are all the train cases, remember that we split off 20% as valset
        with open(join(pp_fld, 'percent_annotated_fold.csv'), 'w') as f:
            f.write('fold,num_cases,percent\n')
            for fld in range(len(splits)):
                f.write(f"{fld},{len(splits[fld]['train'])},{len(splits[fld]['train']) / num_train_cases}\n")

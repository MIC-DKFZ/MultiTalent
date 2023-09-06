from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir




if __name__ == '__main__':

    task_name = "Task100_MultiTalent"

    target_base = join(nnUNet_raw_data, task_name)


    # after cropping we need to add the valid labels to each of the cases. Remeber that they are saved in join(target_base, 'cases_valid_labels_tr.pkl')
    cases_have_labels_tr, cases_have_labels_val, cases_have_labels_ts, cases_have_regions_tr, \
    cases_have_regions_val, cases_have_regions_ts = load_pickle(join(target_base, 'cases_have_regions_labels.pkl'))

    cropped_dir = join(os.environ['nnUNet_raw_data_base'], 'nnUNet_cropped_data', task_name)
    pkl_files = [i for i in subfiles(cropped_dir, suffix='.pkl', join=False) if
                 i != 'dataset_properties.pkl' and i != 'intensityproperties.pkl']
    for p in pkl_files:
        content = load_pickle(join(cropped_dir, p))
        content['valid_labels'] = cases_have_labels_tr[p[:-4] + '.nii.gz']
        content['valid_regions'] = cases_have_regions_tr[p[:-4] + '.nii.gz']
        save_pickle(content, join(cropped_dir, p))

    #  same for preprocessed data
    preprocessed_dirs = subdirs(join(preprocessing_output_dir, task_name), join=True)
    for d in preprocessed_dirs:
        pkl_files = [i for i in subfiles(d, suffix='.pkl', join=False) if
                     i != 'dataset_properties.pkl' and i != 'intensityproperties.pkl']
        for p in pkl_files:
            content = load_pickle(join(d, p))
            content['valid_labels'] = cases_have_labels_tr[p[:-4] + '.nii.gz']
            content['valid_regions'] = cases_have_regions_tr[p[:-4] + '.nii.gz']
            save_pickle(content, join(d, p))


    # this task needs a special trainer

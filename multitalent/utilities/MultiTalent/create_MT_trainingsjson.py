import argparse
from cgi import parse
from copy import deepcopy

from batchgenerators.utilities.file_and_folder_operations import *
from multitalent.dataset_conversion.balints_registration import maybe_mkdir_p
from multitalent.paths import nnUNet_preprocessed
from multitalent.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from multitalent.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset, extract_fingerprint_dataset, preprocess
from multitalent.experiment_planning.experiment_planners.isotropic.isotropic_nnunet import nnUNetPlannerResEncLIso1x1x1_znorm
from multitalent.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner



def prepare_MT_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_name', type=str,
                        help="name of the training")
    parser.add_argument('MTid', type=str,
                        help="mt id, must be a new one")
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs that should be used for MT training. Example: 2 4 9 14")
    parser.add_argument('-p', type=str, required=False, default=None,
                        help='[OPTIONAL] Use this flag to specify a custom plans path. Default: ResEncL 1mm cubic target spacing, zscore')
    parser.add_argument('--omit_preprocessing', action='store_true', required=False,
                        help='[OPTIONAL] also do preprocessing')
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument('-np', type=int,required=False, default=4,
                        help="number of processes for preprocessing. Default 4")
    parser.add_argument('-batch_size', type=int,required=False, default=None,
                        help="[OPTIONAL] Target batch size")
    parser.add_argument('-norm', type=str,required=False, default='ZScoreNormalization',
                        help="[OPTIONAL] Normalization schemes. Default: ZScoreNormalization")
    parser.add_argument('-reader', type=str,required=False, default='SimpleITKIOWithReorient',
                        help="[OPTIONAL] Reader/Writer, default SimpleITKIOWithReorient.")


    args = parser.parse_args()
    MT_dataset_name = "Dataset%03.0d" % int(args.MTid) +'_' + args.training_name
    maybe_mkdir_p(join(nnUNet_preprocessed,MT_dataset_name))

    d_list = ["%03.0d" % id for id in args.d]
    write_json({'dataset_ids': d_list}, join(nnUNet_preprocessed,MT_dataset_name, 'datasets.json'))
    if args.p is not None:
        base_plan = load_json(args.p)
    else:
        extract_fingerprint_dataset(args.d[0], check_dataset_integrity=args.verify_dataset_integrity)
        base_plan, _ = plan_experiment_dataset(args.d[0], nnUNetPlannerResEncLIso1x1x1_znorm)

    target_plan_MT = base_plan.copy()
    target_plan_MT['dataset_name'] = MT_dataset_name
    target_plan_MT['image_reader_writer'] = args.reader
    if args.batch_size is not None:
        for config in target_plan_MT.keys():
            target_plan_MT['configurations'][config]['batch_size'] = args.batch_size
    save_json(target_plan_MT, join(nnUNet_preprocessed,MT_dataset_name,  base_plan['plans_name']+'.json'))

    for id in args.d:
        d_name = convert_id_to_dataset_name(id)
        if not isfile(join(nnUNet_preprocessed,"Dataset%03.0d" % id +'_' + d_name, 'nnUNetPlans.json')):
            _ = extract_fingerprint_dataset(int(id), check_dataset_integrity=args.verify_dataset_integrity)
            default_plan, _ = plan_experiment_dataset(int(id), ExperimentPlanner)
        else:
            default_plan = load_json(nnUNet_preprocessed, d_name, 'nnUNetPlans.json')
        new_plan = default_plan.copy()

        #global arghs of the plan
        new_plan['plans_name'] = base_plan['plans_name']
        new_plan['experiment_planner_used'] = base_plan['experiment_planner_used']

        # i do not wanna have different transposed axis. Not doing 2D anyways
        new_plan['transpose_forward'] = base_plan['transpose_forward']
        new_plan['transpose_backward'] = base_plan['transpose_backward']
        new_plan['image_reader_writer'] = args.reader

        #args for the configuration
        configs = list(new_plan['configurations'].keys())
        for config in configs:
            if config == '3d_fullres':
                #hope that covers all important args
                new_plan['configurations'][config]['data_identifier'] = base_plan['configurations'][config]['data_identifier']
                new_plan['configurations'][config]['preprocessor_name'] = base_plan['configurations'][config]['preprocessor_name']
                new_plan['configurations'][config]['batch_size'] = base_plan['configurations'][config]['batch_size']
                new_plan['configurations'][config]['patch_size'] = base_plan['configurations'][config]['patch_size']
                new_plan['configurations'][config]['spacing'] = base_plan['configurations'][config]['spacing']
                new_plan['configurations'][config]['resampling_fn_data'] = base_plan['configurations'][config]['resampling_fn_data']
                new_plan['configurations'][config]['resampling_fn_seg'] = base_plan['configurations'][config]['resampling_fn_seg']
                new_plan['configurations'][config]['resampling_fn_data_kwargs'] = base_plan['configurations'][config]['resampling_fn_data_kwargs']
                new_plan['configurations'][config]['resampling_fn_seg_kwargs'] = base_plan['configurations'][config]['resampling_fn_seg_kwargs']
                new_plan['configurations'][config]['resampling_fn_probabilities'] = base_plan['configurations'][config]['resampling_fn_probabilities']
                new_plan['configurations'][config]['resampling_fn_probabilities_kwargs'] = base_plan['configurations'][config]['resampling_fn_probabilities_kwargs']
                new_plan['configurations'][config]['architecture'] = base_plan['configurations'][config]['architecture']
                for i in range(len(new_plan['configurations'][config]['normalization_schemes'])):
                    new_plan['configurations'][config]['normalization_schemes'][i] = args.norm
                if args.batch_size is not None:
                    new_plan['configurations'][config]['batch_size'] = args.batch_size

            else:
                del new_plan['configurations'][config]

        plans_file = join(nnUNet_preprocessed, d_name,  new_plan['plans_name']+'.json')
        save_json(new_plan, plans_file, sort_keys=False)

        if not args.omit_preprocessing:
            preprocess([id], base_plan['plans_name'], ['3d_fullres'], num_processes=[args.np])



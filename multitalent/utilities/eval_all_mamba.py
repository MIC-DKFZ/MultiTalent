from batchgenerators.utilities.file_and_folder_operations import *
from multitalent.evaluation.evaluate_predictions import compute_metrics_on_folder2

if __name__ == '__main__':
    base = '/dkfz/cluster/gpu/checkpoints/mamba_slumber/'
    # base = '/home/constantin/cluster-checkpoints_deeper/mamba_slumber/'
    datasets = ['003_Liver', '017_AbdominalOrganSegmentation', '027_ACDC', '137_BraTS2021', '220_KiTS2023', '223_AMOS2022postChallenge']
    trainer = ['nnUNetTrainerV2_MedNeXt_L_kernel3__nnUNetPlansv2.1_trgSp_1x1x1', 'nnUNetTrainerV2_MedNeXt_L_kernel5__nnUNetPlansv2.1_trgSp_1x1x1']
    for did in datasets:
        for fold in range(5):
            fold = 'fold_' + str(fold)
            for tr in trainer:
                print(did, tr)
                if did in ['003_Liver', '220_KiTS2023'] and tr == 'nnUNetTrainerV2_MedNeXt_L_kernel5__nnUNetPlansv2.1_trgSp_1x1x1':
                        tr = 'nnUNetTrainerV2_MedNeXt_L_kernel5_lr_1e_4__nnUNetPlansv2.1_trgSp_1x1x1'
                        respath = join(base, 'nnUNet_v1_results/nnUNet/3d_fullres/Task' + did, tr, fold, 'validation_raw')
                        dspath = join(base, 'nnUNet_results/'+ 'Dataset'+ did, 'STUNetTrainer_small__nnUNetPlans__3d_fullres', 'dataset.json')
                        ppath = join(base, 'nnUNet_results/'+ 'Dataset'+ did, 'STUNetTrainer_small__nnUNetPlans__3d_fullres', 'plans.json')
                        gtpath = join('/dkfz/cluster/gpu/data/mamba_slumber/nnUNet_preprocessed/Dataset' + did, 'gt_segmentations')
                        # gtpath = join('/home/constantin/cluster-data_deeper/mamba_slumber/nnUNet_preprocessed/Dataset' + did,
                        #           'gt_segmentations')
                        # print(dspath)

                        compute_metrics_on_folder2(gtpath, respath, dspath, ppath, output_file=join(respath, 'summary_final.json'), num_processes=8, chill=True)
                else:
                    respath = join(base, 'nnUNet_v1_results/nnUNet/3d_fullres/Task' + did, tr, fold, 'validation_raw')
                    dspath = join(base, 'nnUNet_results/' + 'Dataset' + did, 'STUNetTrainer_small__nnUNetPlans__3d_fullres',
                                  'dataset.json')
                    ppath = join(base, 'nnUNet_results/' + 'Dataset' + did, 'STUNetTrainer_small__nnUNetPlans__3d_fullres',
                                 'plans.json')
                    gtpath = join('/dkfz/cluster/gpu/data/mamba_slumber/nnUNet_preprocessed/Dataset' + did,'gt_segmentations')
                    # gtpath = join('/home/constantin/cluster-data_deeper/mamba_slumber/nnUNet_preprocessed/Dataset' + did,
                    #               'gt_segmentations')
                    compute_metrics_on_folder2(gtpath, respath, dspath, ppath, join(respath, 'summary_final.json'), chill=True)

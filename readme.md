# MultiTalent: A Multi-Dataset Approach to Medical Image Segmentation


<p align="center"> 
<img src="./overview_figure.svg" height="500" title="MultiTalent overview">
</p> <br /> 

Please cite the following work if you find this model useful for your research:

    Ulrich, C., Isensee, F., Wald, T., Zenk, M., Baumgartner, M., & Maier-Hein, K.(2023). 
    MultiTalent: A Multi-Dataset Approach to Medical Image Segmentation. arXiv preprint arXiv:2303.14444.

Please also cite the following work if you use this pipeline for training:

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). 
    nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.

# Requirements

This repository is based on [nnU-Net V1](https://github.com/MIC-DKFZ/nnUNet). 
Please get familiar with it first. 

# Installation
The repository can be cloned and installed using the following commands.

```bash
git clone https://github.com/MIC-DKFZ/MultiTalent.git MultiTalent
cd MultiTalent
pip install -U .
```
Set the Paths of your enviroment according to [nnU-Net V1](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/setting_up_paths.md)

**Note: We are working on an update to nnU-NetV2.**  <br />
In nnU-Net V1, the project was still very experimental. In V2 we will make the project more user-friendly. 

# Apply MultiTalent
You can download a trained Multitalent (U-Net and Residual U-Net) model here using the following command: <br />
`nnUNet_download_pretrained_model Task100_MultiTalent`

After that, you could run the following script for inference: <br />
`CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=1224 --nproc_per_node=1 ./nnunet/inference/predict_MultiTalent.py -i inputpath -o outputpath -m path_to_model_folds`  <br /> 
<p align="center"> 
<img src="./boxplot_mean_final.svg" height="550" title="MultiTalent result">
</p> 

# Fine-tuning MultiTalent
We recommend to preprocess your target dataset as we did for the MultiTalent dataset collection. 



You can fine-tune a MultiTalent model on a new task using the following command: 
```nnUNet_train 3d_fullres nnUNetTrainerV2_warmupsegheads TASK-ID Fold -p plans_name -pretrained_weights  path_to_pretrained_model/model_final_checkpoint.model```<br />
For fine-tuning a residual encoder, use the trainer *nnUNetTrainerV2_warmupsegheads_resenc*.  



# Training with MultiTalent:

**Dataset collection** <br />
In the following, we will give you an instruction how to get and preprocess our partially labelled dataset collection.
You need to download the raw data individually from each dataset origin:<br />
Dataset 1-6: [MSD](http://medicaldecathlon.com/) (only CT datasets) <br />
Dataset 7 & 8: [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217760) <br />
Dataset 9: [BTCV2](https://zenodo.org/record/1169361#.YiDLFnXMJFE) <br />
Dataset 10: [StructSeg](https://structseg2019.grand-challenge.org/) <br />
Dataset 11:  [SegThor](https://competitions.codalab.org/competitions/21145) <br />
Dataset 12: [NIH-Pan](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT) <br />
Dataset 13: [KiTS19](https://kits19.grand-challenge.org/) <br />


After downloading, you have to save the files as expected by nnU-Net in your `nnUNet_raw_data_base` directory and to use the right folder structure ([nnU-Net dataset conversion](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md)). <br /> 
It is important that you choose the same folder names as we did: <br />
    _'Task003_Liver' <br />
    'Task006_Lung' <br />
    'Task007_Pancreas' <br />
    'Task008_HepaticVessel' <br />
    'Task009_Spleen' <br />
    'Task010_Colon' <br />
    'Task017_AbdominalOrganSegmentation' <br />
    'Task046_AbdOrgSegm2' <br />
    'Task051_StructSeg2019_Task3_Thoracic_OAR' <br />
    'Task055_SegTHOR' <br />
    'Task062_NIHPancreas' <br />
    'Task064_KiTS_labelsFixed' <br />
    'Task018_PelvicOrganSegmentation' <br />_ <br />




**Preprocessing for training**
1. First, we need to generate our raw dataset for the multi-class training. Run the following script to copy all the images in the right folder, convert the labelmaps, generate the dataset.jason file and transpose the images if needed. <br />
`/nnunet/dataset_conversion/Task100_MultiTalent.py` It can take around 3 hours, depending on your system.
If you want to extend or change the base dataset collection, you would need to adapt this file!

2. Now, we can use the nn-UNet preprocessing function with a specialized preprocessor: <br />
`nnUNet_plan_and_preprocess -t 100 -pl3d ExperimentPlanner3D_v21_MultiTalent -pl2d None -tf 16 --verify_dataset_integrity -overwrite_plans_identifier multitalent_bs4` Again, this takes some time. <br />
This also generates a training plan file that we need for the following network training. By default, this plan generates a batchsize of 2. It is very easy to change the batchsize (see [extending nnU-Net](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md)), but you could also use the plans that we provide. 


4. We are almost done, but we need to add the information about the valid labels for each image to the _.pkl_ files: <br />
`python /nnunet/dataset_conversion/Task100_MultiTalent_addregions.py` This takes only a few seconds. <br /><br />




**Training of the Multi-Class network** <br />
First, you should take a look at the nnU-Net V1 distributed training instructions. 
To train a MultiTalent network, run the following command: <br />
`CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=1234 --nproc_per_node=2 ./nnunet/run/run_training_DDP.py 3d_fullres MultiTalent_trainer_ddp 100  0 -p MultiTalent_bs4 --dbs`









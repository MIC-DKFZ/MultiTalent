# Welcome to the **new MultiTalent**!

**Train your own MultiTalent:**

MultiTalent allows you to train with ANY 3D dataset, as long it is compatible with nnU-Netv2.
So, please make yourself familiar with [nnU-Netv2](https://github.com/MIC-DKFZ/nnUNet), and prepare your datasets accordingly. 

Follow the steps here to get started (identical to nnunet):
  ```bash
  git clone https://github.com/MIC-DKFZ/MultiTalent.git
  cd multitalent
  pip install -e .
  ```
Remember, to also set the [nnU-Net paths](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).
After that, prepare the datasets you want to train with, as expected by nnunet: [Dataset conversion](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

Now, we want to preprocess all data similary as expected for the MultiTalent combined training. depending on the number and sice of dataset that are selected, this can take some time. 

>prepare_MT_training MT_name MT_id -d nnunet_ids --verify_dataset_integrity

`MT_name`: Must be a new nnU-Net dataset name \
`MT_id`: Must be a new nnU-Net dataset ID \
`nnunet_ids`: Is a list of nnU-Net dataset IDs that will be used for the MultiTalent training. e.g.  `-d 3 6 7 8 9 10` for all Medical decathlon CT datasets \
`-p`: (Optional) Path to a plans.json file that is used for the MultiTalent training. We recommend to just use the provided `nnUNetResEncUNetLPlansIso1x1x1.json`.
    If not set, the plan resulting from the experiment planner for a ResEncL with fixed target spacing 1x1x1 for the first dataset that is provided will be used for all others.  


>multitalent_train experimentID 3d_fullres fold -p targeplansname -tr MultiTalent_trainer

When training with more than 500images in total, we recommend to increase batch size and training length.

**Fine-Tuning**
ToDo


**Run Inference**

ToDo



Please cite the following work if you find this model useful for your research. If you want to reproduce the paper results, checkout the corresponding branch (MultiTalentV1). :

    Ulrich, C., Isensee, F., Wald, T., Zenk, M., Baumgartner, M., Maier-Hein, K.H. (2023). 
    MultiTalent: A Multi-dataset Approach to Medical Image Segmentation. In: Greenspan, H., et al. 
    Medical Image Computing and Computer Assisted Intervention – MICCAI 2023.

Please also cite the following work if you use this pipeline for training:

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). 
    nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.



# assumes you are in the nnunet repo!

# prepare raw datasets
python multitalent/dataset_conversion/datasets_for_integration_tests/Dataset999_IntegrationTest_Hippocampus.py
python multitalent/dataset_conversion/datasets_for_integration_tests/Dataset998_IntegrationTest_Hippocampus_ignore.py
python multitalent/dataset_conversion/datasets_for_integration_tests/Dataset997_IntegrationTest_Hippocampus_regions.py
python multitalent/dataset_conversion/datasets_for_integration_tests/Dataset996_IntegrationTest_Hippocampus_regions_ignore.py

# now run experiment planning without preprocessing
multitalent_plan_and_preprocess -d 996 997 998 999 --no_pp

# now add 3d lowres and cascade
python multitalent/tests/integration_tests/add_lowres_and_cascade.py -d 996 997 998 999

# now preprocess everything
multitalent_preprocess -d 996 997 998 999 -c 2d 3d_lowres 3d_fullres -np 8 8 8  # no need to preprocess cascade as its the same data as 3d_fullres

# done
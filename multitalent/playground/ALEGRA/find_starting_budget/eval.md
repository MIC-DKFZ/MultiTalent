981, 982, 983, 984 are the datasets with patches as annotations. They contain 21, 50, 100 and 200 patches, respectively.
(distributed over the 21 train cases we have)

Patches are sampled with the following strategy:

Baseline is a model trained on fewer train cases in a fully annotated setting Dataset 980. Folds:
- 0, 1, 2: 1 train case
- 3, 4, 5: 3 train cases
- 6, 7, 8: 5 train cases
- 9, 10, 11: 10 train cases
- 12, 13, 14: 15 train cases
- 'all': all train cases

See [Dataset980ff_ALEGRA_findStartingBudget.py](../../../dataset_conversion/Dataset980ff_ALEGRA_findStartingBudget.py) for dataset conversion.

980-981 are already resampled to the target spacing in order to avoid problems with patch sizes resulting from 
resizing. This is consistent with the projected nnActive behavior

Evaluation of all models is done on the validation set of 980. Those are the ~100 images that were never manually 
annotated. We use the previously generated segmentation masks (result from HI collaboration) as GT. This should be 
precise enough to measure model performance.


Results are here: https://docs.google.com/spreadsheets/d/1mIdfIE8YwBQ15abrZfy7u1xLqfdZpZGqNC6u8hHeDEg/edit?usp=sharing

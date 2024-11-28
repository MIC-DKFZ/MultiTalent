from multitalent.playground.ALEGRA.evaluate_sparse_slicewise_annotations.configuration import *

if __name__ == '__main__':
    with open('run_commands.sh', 'w') as f:
        # low num train cases
        for d in datasets:
            gpu_mem = 1 if d == 980 else 33 # this will move brats and amos jobs to a100 nodes with nice cpus. otherwise we die of CPU starvation
            trainer = 'nnUNetTrainer' if not d == 980 else 'nnUNetTrainer_airwayAug_new'
            for pa in range(len(percent_of_cases_annotated)):
                for i in range(num_runs):
                    f.write(f"bsub -q gpu -gpu num=1:j_exclusive=yes:gmem={gpu_mem}G -L /bin/bash "
                            f"\"source ~/load_env_cluster4.sh && nnUNet_results={results_folder} "
                            f"nnUNet_preprocessed={nnUNet_preprocessed_alegra} "
                            f"multitalent_train {d} 3d_fullres {pa * num_runs + i} -t {trainer} --disable_checkpointing\"\n")
        # sparse annotations
        for d in datasets:
            gpu_mem = 1 if d == 980 else 33 # this will move brats and amos jobs to a100 nodes with nice cpus. otherwise we die of CPU starvation
            trainer = 'nnUNetTrainer' if not d == 980 else 'nnUNetTrainer_airwayAug_new'
            for c in new_configurations.keys():
                fl = len(percent_of_cases_annotated) * num_runs - 1
                f.write(f"bsub -q gpu -gpu num=1:j_exclusive=yes:gmem={gpu_mem}G -L /bin/bash "
                        f"\"source ~/load_env_cluster4.sh && nnUNet_results={results_folder} "
                        f"nnUNet_preprocessed={nnUNet_preprocessed_alegra} "
                        f"multitalent_train {d} {c} {fl} --disable_checkpointing\"\n")


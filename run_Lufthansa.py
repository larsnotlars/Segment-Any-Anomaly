import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == "__main__":

    pool = Pool(processes=1)

    dataset_list = ["lufthansa"]
    gpu_indx = 0

    for dataset in dataset_list:
        classes = dataset_classes[dataset]
        for cls in classes[:]:
            sh_method = (
                f"python eval_SAA.py "
                f"--dataset {dataset} "
                f"--class-name {cls} "
                f"--batch-size {1} "
                f"--root-dir ./result "
                f"--cal-pro False "
                f"--use-cpu 1 "
                f"--gpu-id {gpu_indx} "
                f"--eval-resolution 2000"
            )
            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()

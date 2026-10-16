# Adaptive-HAR
This project implements adaptive inference for wearable Human Activity Recognition (HAR) using accelerometer time-series data from the CAPTURE-24 dataset. The objective is to balance predictive performance and computational cost by adapting the amount of computation performed for each inference according to the available battery budget and the difficulty of the input. Instead of using a single fixed depth model for every input, the system combines structured pruning with a shared-weight early-exit architecture. A battery-aware policy determines the maximum permitted inference depth, while confidence thresholds determine whether an input can exit early.

## Main contributions

- A ResNet1D baseline for wearable human activity recognition.
- Structured channel pruning using DepGraph and magnitude-based importance.
- A shared-weight early-exit model with three exit points.
- A battery aware inference policy with low, medium, and high operating conditions.
- Evaluation using accuracy, macro F1-score, balanced accuracy, FLOPs, parameter count, model size, and latency.
- Pareto-frontier analysis to compare predictive performance against computational cost.

### Assumptions:
- Working environment is Das5
- Using conda as environment manager
- Navigating into the correct directory (example: to run prune.sh you need to be in the prune directory)

## How to run 

### **Start by cloning the repo to your local machine**
````
Git clone https://github.com/Isabella-Anthony/Adaptive-HAR.git
````
### **Setup your environment using conda**
````
source /var/scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda create adaptivity
````
### **Download all Requirements**
````
pip install -r requirements.txt
````

## Dataset

### Download the CAPTURE-24 dataset, make sure the folder name remains as "capture24"
````
https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001
````


## Filestructure

````
adaptivty/
├── adaptivity_mechanism/
│   ├── controller.py
│   └── inference.py
├── evaluate/
│   ├── eval_adaptive.py
│   ├── eval_baseline.py
│   ├── eval_prune.py
│   ├── plots/
│   │   ├── plot_battery_simulation.py
│   │   ├── plot_pareto.py
│   │   └── plot.sh
│   ├── send_eval_adaptivity.sh
│   ├── send_eval_all.sh
│   ├── send_eval_baseline.sh
│   └── send_eval_prune.sh
├── preprocess/
│   ├── prepros_das5.py
│   └── send_preprocessing.sh
├── prune/
│   ├── prune.py
│   └── prune.sh
├── training/
│   ├── send_fit_early_exit.sh
│   ├── send_train_adaptive.sh
│   ├── send_train_baseline.sh
│   ├── tfit_early_exit.py
│   ├── train_adaptive.py
│   └── train_baseline.py
├── utility/
│   ├── early_exit_model.py
│   ├── loaders.py
│   ├── model.py
│   └── utils.py
└── zresults/
    ├── baseline_results/
    │   ├── baseline_confusion_matrix.png
    │   ├── baseline_metrics.json
    │   └── resnet1d_baseline_best.pt
    ├── early_exit_results/
    │   ├── early_exit_ratio30_best.py
    │   ├── early_exit_ratio30_finetuned_best.py
    │   ├── early_exit_ratio30_finetuned_comprehensive_metrics.json
    │   ├── exit_distribution_breakdown.png
    │   └── pareto_frontier_with_baseline.png      
    └── pruning_results/
        ├── resnet1d_pruned_ratio30_confusion_matrix.png
        ├── resnet1d_pruned_ratio30_metrics.json
        └── resnet1d_pruned_ratio30.pt````

````
## End-to-End Workflow

To run the complete project from training until plotting. Please run the scripts in the following order as stated below. If you wish to only evaluate the trained models, please see next section.

* make sure you are in the directory that the file is located in, otherwise you will run into errors!

 1.Preprocess the CAPTURE-24 dataset by running the following commands:
````
sbatch send_preprocessing.sh 
````
2.Train and evaluate the baseline ResNet1D model by running the following commands:
````
sbatch send_train_baseline.sh
````
````
sbatch send_eval_baseline.sh
````
3.Apply structured pruning using DepGraph, fine-tune and evaluate the pruned model by running the following commands:
````
sbatch prune.sh
````
````
sbatch send_eval_prune.sh
````
4.Add and fine-tune the shared-weight early-exit by running the following commands:
````
sbatch send_train_adaptive.sh
````
````
sbatch send_fit_early_exit.sh
````
 5.Evaluate the adaptive inference policies at low, medium, and high battery levels by running the following commands:
````
sbatch send_eval_adaptivity.sh
````
6.Plot all the results and accuracy-versus-FLOPs Pareto frontier by running the following commands:
````
sbatch plot.sh
````

## Evaluation Workflow

To evaluate the results from our adaptive solution. Please run the scripts in the following order as stated below.

* make sure you are in the directory that the file is located in, otherwise you will run into errors!

 1.Preprocess the CAPTURE-24 dataset by running the following commands:
````
sbatch send_preprocessing.sh 
````
 2.Script to evaluate baseline, pruned, and adaptive model:
````
sbatch send_eval_all.sh
````
3.Plot all the results and accuracy-versus-FLOPs Pareto frontier by running the following commands:
````
sbatch plot.sh
````



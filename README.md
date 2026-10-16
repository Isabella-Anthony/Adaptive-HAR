# Adaptive-HAR
Adaptive Inference on wearables for human activity recognition. 

Assumptions:
- Working environment is Das5

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

### Filestructure

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
### Workflow

Run the project scripts in the following order:

## 1.Preprocess the CAPTURE-24 dataset by running the following commands:
````
sbatch send_preprocessing.sh 
````
## 2.Train and evaluate the baseline ResNet1D model by running the following commands:
````
sbatch send_train_baseline.sh
````
````
sbatch send_eval_baseline.sh
````
## 3.Apply structured pruning using DepGraph and fine-tune the pruned model by running the following commands:
````
sbatch prune.sh
````
````
tba
````
## ** 4.Add and fine-tune the shared-weight early-exit by running the following commands:
````
tba
````
````
tba
````
## 5.Evaluate the adaptive inference policies at low, medium, and high battery levels by running the following commands:
````
tba
````
````
tba
````
## 6.Generate all the results tables and accuracy-versus-FLOPs Pareto frontier by running the following commands:
````
tba
````
````
tba
````


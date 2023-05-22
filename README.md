# KDBNet
## Installation
```bash
git clone https://github.com/luoyunan/KDBNet.git
cd KDBNet
export PYTHONPATH=$PWD:$PYTHONPATH
```
## Dependencies
This package is tested with Python 3.9 and CUDA 11.6 on Ubuntu 20.04, with access to an Nvidia A40 GPU (48GB RAM), AMD EPYC 7443 CPU (2.85 GHz), and 512G RAM. Run the following to create a conda environment and install the required Python packages (modify `pytorch-cuda=11.6` according to your CUDA version). 
```bash
conda create -n kdbnet python=3.9
conda activate kdbnet

conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c conda-forge uncertainty-toolbox rdkit pyyaml
```
Running the above lines of `conda install` should be sufficient to install all  KDBNet's required packages (and their dependencies). Specific versions of the packages we tested were listed in `requirements.txt`.
## Quick Example
1. Download example data (~120MB) from Dropbox.
    ```bash
    wget https://www.dropbox.com/s/owc45bzbfn05ix4/data.tar.gz
    tar -xf data.tar.gz
    ```
2. Run the example code in `scripts/`. The following script trains and evaluates a KDBNet model using the Davis dataset of kinase-drug binding affinity. The script randomly splits 70% as training data, 10% as validation data, and 20% as test data.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_example.py --task davis --n_epochs 100 --output_dir ../output/davis --save_prediction
    ```
    After the training and testing are finished, the evaluation metrics will be printed in the tarminal. The predicted binding affinity values will be saved as a TSV file in the output directory, with the following format
    ```
    drug	    protein	    y_true	    y_pred
    11667893	MKK7	    5	        5.15088
    ...         ...         ...         ...
    ```
## Other usages
1. Dataset split. By default, the dataset is randomly split to train/valid/test with a ratio 0.7/0.1/0.2. We also support dataset split by drug, protein, or both, such that the model is tested on unseen proteins, unseen drugs or both. To use this option, set the argument `--split_method` using `drug` or `both` for the `--split_method` method. For example, to test on unseen drugs, run the following script
    ```
    CUDA_VISIBLE_DEVICES=0 python run_example.py --task davis --n_epochs 100 --split_method drug
    ```
    We also support sequence identity-based split such that the model is tested on unseen proteins with sequence identity lower 50% (`--split_method seqid`). To use other sequence identity cutoff, first run MMseqs2 (`cluster` function) to generate the clustering file, and change the initialize the dataset class (`DAVIS` or `KIBA` in `dta.py`) with the file path, e.g., `dataset = DAVIS(mmseqs_seq_cluster_file='mmseqs_cluster.tsv')`.
2. Ensemble. To ensemble multiple models, use `--n_ensembles` argument. For example, to ensemble 5 models, run
    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_example.py --task davis --n_epochs 100 --n_ensembles 5 --output_dir ../output/davis_ens --save_prediction
    ```
    In the output TSV file, the `y_pred` column is the average of the predictions by the 5 models, and the predictions given by individual model is listed in the `y_pred_0`, `y_pred_1`, ..., `y_pred_4` columns.
3. Uncertainty estimation. To estimate the uncertainty of the model, use `--uncertainty` argument. When `--uncertainty` is set to `True`, the number of ensembles should be greater than 1. The model will output the mean and standard deviation of the prediction.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_example.py --task davis --n_epochs 100 --n_ensembles 5 --uncertainty --output_dir ../output/davis_unc --save_prediction
    ```
    In the output TSV file, there will be a `y_std` column, which is the standard deviation of the predictions by the 5 models and can be used as the uncertainty estimates.
4. Parallel training. To train ensemble models in parallel, use `--parallel` argument. For example, to train 5 models in parallel, run
    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_example.py --task davis --n_epochs 100 --n_ensembles 5 --uncertainty --parallel
    ```
    You can train a separate model in the ensembles on a single GPU by setting `CUDA_VISIBLE_DEVICES=0,1,2,3,4`. When you have fewer GPUs than the number of ensembles, multiple models may be trained on a single GPU. For example, if you have 2 GPUs and want to train an ensemble of 5 models, you can set `CUDA_VISIBLE_DEVICES=0,1`, and the 3 models will be trained on GPU 0, and the other 2 models will be trained on GPU 1.
5. Uncertainty recalibration. Use `--recalibrate` argument to perform uncertainty recalibration. For example, to ensemble 5 models and perform uncertainty recalibration, run
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2 python run_example.py --task davis --n_epochs 100 --n_ensembles 5 --uncertainty --parallel --recalibrate --output_dir ../output/davis_recal --save_prediction
    ```
    In the output TSV file, there will be a `y_std_recalib` column, which is the recalibrated uncertainty.

## Description of data files
1. `data/DAVIS/` is the Davis dataset of kinase-drug binding affinity. The `davis_data.tsv` contains the binding affinity data with the following format
    ```
    drug	protein	Kd	    y
    5291	AAK1	10000	5
    ...     ...     ...     ...
    ```
    where `drug` is the PubChem Compound ID (CID) of the drug, `protein` is the kinbase name, `Kd` is the binding affinity in nM, and `y` is the log-transformed binding affinity (see paper). The `davis_protein2pdb.yaml` file contains the mapping from a kinase name to its representative PDB structure ID. The `davis_cluster_id50_cluster.tsv` is the [clustering output file](https://github.com/soedinglab/MMseqs2/wiki#cluster-tsv-format) of the MMseqs2 clustering algorithm with a sequence identity cutoff 50% (the first column contains the representative sequences and the second column contains cluster members).
2. `data/KIBA/` is the KIBA dataset of kinase-inhibitor binding affinity. The files in this directory are similar to those in `data/DAVIS/`. In the `kiba_data.tsv` file, the `drug` column contains CHEMBL IDs of the drugs, and the `protein` column contains UniProt IDs of the kinases.
3. `data/structure` contains several structure files. 
    - The `pockets_structure.json` contains the PDB structure data of representative kinase structures. The file is in JSON format where the key is the PDB ID, and the value is the corresponding PDB structure data in a dictionary format, including the following fields: `name` (kinase name), `UniProt_id`, `PDB_id`, `chain` (chain ID in the PDB structure), `seq` (pocket protein sequence), `coords` (coordinates of the N, CA, C, O atoms of residues in the pocket). The `coords` is a dictionary with four fields, and each is a list of xyz coordinates of the N/CA/C/O atom in each residue, i.e., `coords={'N':[[x, y, z], ...]], 'CA': [...], 'C': [...], O: [...]}`
    - The `davis_moil3d_sdf` and `kiba_moil3d_sdf` are diretories that contain the 3D structure (SDF format) of molecules in the Davis and KIBA datasets.
4. `data/esm1b` contains the pre-computed protein sequence embeddings by the ESM-1b model. The embeddings were saved as PyTorch tensors in the `.pt` format. 

## Contact
Please submit GitHub issues or contact Yunan Luo (luoyunan[at]gmail[dot]com) for any questions related to the source code.
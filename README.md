# CLEAN-Contact

This repository contains the code and data for the paper "CLEAN-Contact: Contrastive learning enabled enzyme function prediction with contact map".

## Installation

Python == 3.10.13, PyTorch == 2.1.1, torchvision == 0.16.1;
fair-esm == 2.0.0, pytorch-cuda == 12.1

Follow https://pytorch.org/ to install PyTorch and torchvision with CUDA. 

```bash
git clone https://github.com/PNNL-CompBio/CLEAN-contact-for-public.git
cd CLEAN-contact-for-public
conda create -n clean-contact python=3.10 -y
conda activate clean-contact
python -m pip install fair-esm==2.0.0
python build.py install
git clone https://github.com/facebookresearch/esm.git
```

## Prepare data

First create required folders:

```bash
from src.CLEAN.utils import ensure_dirs
ensure_dirs()
```

Download the precomputed embeddings and distance map for both training and test data from [here](localhost) and put them in the `data` folder.

To extract sequence representations and structure representations for your own data, first prepare the protein structures in PDB format under `<pdb-dir>` and the dataset in csv format at `<csv-file>`. 

For example, your `<csv-file>` is `data/split100_reduced.csv`. Then run the following commands: 

```bash
python extract_structure_representation.py \
    --input data/split100_reduced.csv \
    --pdb-dir <pdb-dir> 
```

```python
python

>>> from src.CLEAN.utils import csv_to_fasta, retrive_esm1b_embedding

>>> csv_to_fasta('data/split100_reduced.csv', 'data/split100_reduced.fasta')

>>> retrive_esm1b_embedding('split100_reduced')
```

## Inference

If your dataset is in `csv` format, you can use the following command to inference the model:

```bash
python inference.py \
    --train-data split100_reduced \
    --test-data <test-data> \
    --gmm <gmm> \
    --method <method>
```

Replace `<test-data>` with your test data name, `<gmm>` with the list of fitted Gaussian Mixture Models (GMMs) and `<method>` with the `maxsep` or `pvalue`.

If you provide `<gmm>`, the model will use the fitted GMMs to compute prediction confidence. 

Run `python extract_confidence_result.py` and `python print_prediction_confidence_results.py` to extract and print the prediction confidence results, respectively, to reproduce results in Fig. S4-6.

We provide the fitted GMMs based on `maxsep` at `gmm_test/gmm_lst.pkl`. 

If your dataset is in `fasta` format, you can use the following command to inference the model:

```bash
python inference_fasta.py \
    --train-data split100_reduced \
    --fasta <fasta-file> \
    --gmm <gmm> \
    --method <method>
```

Performance metrics measured by Precision, Recall, F-1, and AUROC will be printed out. Per sample predictions will be saved in `results` folder.

## Training

Sequences whose EC number has only one sequence are required to mutated to generate positive samples. We provide the mutated sequences in `data/split100_reduced_single_seq_ECs.csv`. To get your own mutated sequences, run the following command:

```python
python

>>> from src.CLEAN.utils import mutate_single_seq_ECs

>>> mutate_single_seq_ECs('split100_reduced')
```

```bash
python mutate_conmap_for_single_EC.py \
    --fasta data/split100_reduced_single_seq_ECs.fasta 
```

```python
python

>>> from src.CLEAN.utils import fasta_to_csv, merge_sequence_structure_emb

>>> fasta_to_csv('data/split100_reduced_single_seq_ECs.fasta', 'data/split100_reduced_single_seq_ECs.csv')

>>> merge_sequence_structure_emb('split100_reduced_single_seq_ECs')
```

To train the model mentioned in the main text (`addition` model), modify arguments in `train-split100-reduced-resnet50-esm2-2560-addition-triplet.sh` and run the following command:

```bash
./train-split100-reduced-resnet50-esm2-2560-addition-triplet.sh
```

To train models with the other combinations (`contact_1` and `contact_2`), modify arguments in `train-split100-reduced-resnet50-esm2-2560-contact_1-triplet.sh` and `train-split100-reduced-resnet50-esm2-2560-contact_2-triplet.sh`, respectively, and run the following command:

```bash
./train-split100-reduced-resnet50-esm2-2560-contact_1-triplet.sh
./train-split100-reduced-resnet50-esm2-2560-contact_2-triplet.sh
```

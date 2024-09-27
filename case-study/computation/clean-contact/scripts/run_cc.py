import subprocess
import os
from src.CLEAN.utils import fasta_to_csv, retrieve_esm2_embedding, merge_sequence_structure_emb
from src.CLEAN.infer import *

# inputs_list contains original fasta, with annotations still in headers, starts with 1942 ids
inputs_list = ["data/med4_uniprotkb_proteome_UP000001026_2024_01_13"]

for i in inputs_list:
	## preinference
	subprocess.run(['echo', 'about to run pre-inference steps'])
	# if needed, creates fasta with only ids in the headers before getting the embeddings
	# these fasta ids need to match the id within the pdb file names, such as 'AF-{id}-F1-model_v4'
	retrieve_esm2_embedding(os.path.basename(i))
	only_ids_name = os.path.basename(i)+'_only_ids_in_headers'
	# fasta_to_csv will only add id from the fasta to the csv if the id is in the pdb folder
	# so here our number of ids drops to 1939
	fasta_to_csv(only_ids_name)
	subprocess.run(['echo', 'ran fasta_to_csv and retrieve_esm2_embedding, now run extract_structure_representation.py'])
	struct_in=i+'_only_ids_in_headers.csv'
	subprocess.run(['python', 'extract_structure_representation.py', '--input', struct_in, '--pdb-dir', 'my_pdb', '--device', 'cuda:0'])
	subprocess.run(['echo', 'now run merge_sequence_structure_emb'])
	merge_sequence_structure_emb(only_ids_name)
	## inference
	# these steps grab the csvs
	subprocess.run(['echo', 'now run inference using maxsep with gmm'])
	subprocess.run(['python', 'inference.py', '--train-data', 'split100_reduced', '--test-data', only_ids_name,  '--method', 'maxsep', '--gmm', './gmm_test/gmm_lst.pkl'])
	infer_maxsep(train_data='split100_reduced', test_data=only_ids_name, model_name='split100_reduced_resnet50_esm2_2560_addition_triplet',gmm='./gmm_test/gmm_lst.pkl', report_metrics=False, pretrained=False)
	subprocess.run(['echo', 'now run inference using pvalue with gmm'])
	subprocess.run(['python', 'inference.py', '--train-data', 'split100_reduced', '--test-data', only_ids_name, '--method', 'pvalue'])
	infer_pvalue(train_data='split100_reduced', test_data=only_ids_name, model_name='split100_reduced_resnet50_esm2_2560_addition_triplet', gmm='./gmm_test/gmm_lst.pkl', report_metrics=False, pretrained=False)

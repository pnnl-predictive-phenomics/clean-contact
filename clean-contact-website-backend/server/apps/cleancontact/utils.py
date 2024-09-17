from transformers import EsmModel, AutoTokenizer
from Bio import PDB
from Bio.PDB.Polypeptide import three_to_index, index_to_one
import biotite.structure as bs
from biotite.structure.io.pdb import PDBFile, get_structure
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import csv
import pickle

def sequence_from_pdb(pdb_file: str):

    parser = PDB.PDBParser()
    structure = parser.get_structure("PDB_structure", pdb_file)

    sequence = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    sequence.append(index_to_one(three_to_index(residue.resname)))

    return "".join(sequence)

def get_esm_x(seq: str):

    pretrained_path = "/var/www/.cache/huggingface/esm2_t36_3B_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model = EsmModel.from_pretrained(pretrained_path)

    inputs = tokenizer(seq, return_tensors="pt")
    outputs = model(**inputs).last_hidden_state[0, :]

    return outputs.mean(dim=0)

def extend(a, b, c, L, A, D):
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])

def contacts_from_pdb(
    structure: bs.AtomArray,
    distance_threshold: float = 8.0,
    chain: Optional[str] = None,
) -> np.ndarray:
    mask = ~structure.hetero
    if chain is not None:
        mask &= structure.chain_id == chain

    N = structure.coord[mask & (structure.atom_name == "N")]
    CA = structure.coord[mask & (structure.atom_name == "CA")]
    C = structure.coord[mask & (structure.atom_name == "C")]

    Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143)
    dist = squareform(pdist(Cbeta))
    
    contacts = dist < distance_threshold
    contacts = contacts.astype(np.int64)
    contacts[np.isnan(dist)] = -1
    return contacts

def get_con_x(
    pdb_file: Optional[str] = None,
):

    structure = get_structure(PDBFile.read(pdb_file))[0]
    contact_map = contacts_from_pdb(structure)
    
    cmap = np.array([contact_map, contact_map, contact_map])

    with torch.no_grad():
        model = torchvision.models.get_model("resnet50", weights="IMAGENET1K_V2")
        model.fc = nn.Identity()
        model.eval()

        emb = model(torch.tensor(cmap).unsqueeze(0).to(dtype=torch.float32)).cpu().squeeze(0)

    return emb


def model_embedding_test(esm_x, con_x, model):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    then inferenced with model to get model embedding
    '''
    
    model_emb = model(esm_x, con_x)
    return model_emb

def get_dist_map_test(model_emb_train, model_emb_test,
                      ec_id_dict_train, test_id,
                      device, dtype, dot=False):
    '''
    Get the pair-wise distance map for test queries and train EC cluster centers
    map is of size of (N_test_ids, N_EC_train)
    '''
    print("The embedding sizes for train and test:",
          model_emb_train.size(), model_emb_test.size())
    # get cluster center for all EC appeared in training set
    cluster_center_model = get_cluster_center(
        model_emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), model_emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    # calculate distance map between n_query_test * total_ec_n (training) pairs
    ids = [test_id]
    print(f'Calculating eval distance map, between {len(ids)} test ids '
          f'and {total_ec_n} train EC cluster centers')
    if dot:
        eval_dist = dist_map_helper_dot(ids, model_emb_test, ecs, model_lookup)
    else:
        eval_dist = dist_map_helper(ids, model_emb_test, ecs, model_lookup)
    return eval_dist

def get_random_nk_dist_map(emb_train, rand_nk_emb_train,
                           ec_id_dict_train, rand_nk_ids,
                           device, dtype, dot=False):
    '''
    Get the pair-wise distance map between 
    randomly chosen nk ids from training and all EC cluster centers 
    map is of size of (nk, N_EC_train)
    '''
    cluster_center_model = get_cluster_center(emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    if dot:
        random_nk_dist_map = dist_map_helper_dot(
            rand_nk_ids, rand_nk_emb_train, ecs, model_lookup)
    else:
        random_nk_dist_map = dist_map_helper(
            rand_nk_ids, rand_nk_emb_train, ecs, model_lookup)
    return random_nk_dist_map

def dist_map_helper_dot(keys1, lookup1, keys2, lookup2):
    dist = {}
    lookup1 = F.normalize(lookup1, dim=-1, p=2)
    lookup2 = F.normalize(lookup2, dim=-1, p=2)
    for i, key1 in enumerate(keys1):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm**2
        #dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist


def dist_map_helper(keys1, lookup1, keys2, lookup2):
    dist = {}
    for i, key1 in enumerate(keys1):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist

def get_cluster_center(model_emb, ec_id_dict):
    cluster_center_model = {}
    id_counter = 0
    with torch.no_grad():
        for ec in list(ec_id_dict.keys()):
            ids_for_query = list(ec_id_dict[ec])
            id_counter_prime = id_counter + len(ids_for_query)
            emb_cluster = model_emb[id_counter: id_counter_prime]
            cluster_center = emb_cluster.mean(dim=0)
            cluster_center_model[ec] = cluster_center.detach().cpu()
            id_counter = id_counter_prime
    return cluster_center_model

def get_ec_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id

def random_nk_model(id_ec_train, ec_id_dict_train, emb_train, n=10, weighted=False):
    ids = list(id_ec_train.keys())
    nk = n * 1000
    if weighted:
        P = []
        for id in id_ec_train.keys():
            ecs_id = id_ec_train[id]
            ec_densities = [len(ec_id_dict_train[ec]) for ec in ecs_id]
            # the prob of calling this id is inversely prop to 1/max(density)
            P.append(1/np.max(ec_densities))
        P = P/np.sum(P)
        random_nk_id = np.random.choice(
            range(len(ids)), nk, replace=True, p=P)
    else:
        random_nk_id = np.random.choice(range(len(ids)), nk, replace=False)

    random_nk_id = np.sort(random_nk_id)
    chosen_ids = [ids[i] for i in random_nk_id]
    chosen_emb_train = emb_train[random_nk_id]
    return chosen_ids, chosen_emb_train

def write_pvalue_choices(df, random_nk_dist_map, p_value=1e-5, gmm=None):
    # out_file = open(csv_name + '_pvalue.csv', 'w', newline='')
    # csvwriter = csv.writer(out_file, delimiter=',')
    # out_file_confidence = open(csv_name + '_pvalue_confidence.csv', 'w', newline='')
    # csvwriter_confidence = csv.writer(out_file_confidence, delimiter=',')
    results = []
    confidence_results = []
    all_test_EC = set()
    nk = len(random_nk_dist_map.keys())
    threshold = p_value*nk
    for col in df.columns:
        ec = []
        ec_confidence = []
        smallest_10_dist_df = df[col].nsmallest(10)
        for i in range(10):
            EC_i = smallest_10_dist_df.index[i]
            # find all the distances in the random nk w.r.t. EC_i
            # then sorted the nk distances
            rand_nk_dists = [random_nk_dist_map[rand_nk_id][EC_i]
                             for rand_nk_id in random_nk_dist_map.keys()]
            rand_nk_dists = np.sort(rand_nk_dists)
            # rank dist_i among rand_nk_dists
            dist_i = smallest_10_dist_df[i]
            rank = np.searchsorted(rand_nk_dists, dist_i)
            if (rank <= threshold) or (i == 0):
                dist_str = "{:.4f}".format(dist_i)
                all_test_EC.add(EC_i)
                ec.append('EC:' + str(EC_i) + '/' + dist_str)
                if gmm != None:
                    gmm_lst = pickle.load(open(gmm, 'rb'))
                    mean_confidence_i, std_confidence_i = infer_confidence_gmm(dist_i, gmm_lst)
                    confidence_str = "{:.4f}_{:.4f}".format(mean_confidence_i, std_confidence_i)
                    ec_confidence.append('EC:' + str(EC_i) + '/' + confidence_str)
            else:
                break
        ec.insert(0, col)
        results.append(ec)
        if gmm != None:
            ec_confidence.insert(0, col)
            confidence_results.append(ec_confidence)
    return results, confidence_results

def write_max_sep_choices(df, first_grad=True, use_max_grad=False, gmm = None):
    # out_file = open(csv_name + '_maxsep.csv', 'w', newline='')
    # out_file_confidence = open(csv_name + '_maxsep_confidence.csv', 'w', newline='')
    # csvwriter = csv.writer(out_file, delimiter=',')
    # csvwriter_confidence = csv.writer(out_file_confidence, delimiter=',')
    results = []
    confidence_results = []
    all_test_EC = set()
    for col in df.columns:
        ec = []
        ec_confidence = []
        smallest_10_dist_df = df[col].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)
        max_sep_i = maximum_separation(dist_lst, first_grad, use_max_grad)
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            if gmm != None:
                gmm_lst = pickle.load(open(gmm, 'rb'))
                mean_confidence_i, std_confidence_i = infer_confidence_gmm(dist_i, gmm_lst)
                confidence_str = "{:.4f}_{:.4f}".format(mean_confidence_i, std_confidence_i)
                ec_confidence.append('EC:' + str(EC_i) + '/' + confidence_str)
            dist_str = "{:.4f}".format(dist_i)
            all_test_EC.add(EC_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        results.append(ec)
        if gmm != None:
            ec_confidence.insert(0, col)
            confidence_results.append(ec_confidence)
    return results, confidence_results

def maximum_separation(dist_lst, first_grad, use_max_grad):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        max_sep_i = large_grads[-1][opt]
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i

def infer_confidence_gmm(distance, gmm_lst):
    confidence = []
    for j in range(len(gmm_lst)):
        main_GMM = gmm_lst[j]
        a, b = main_GMM.means_
        true_model_index = 0 if a[0] < b[0] else 1
        certainty = main_GMM.predict_proba([[distance]])[0][true_model_index]
        confidence.append(certainty)
    return np.mean(confidence), np.std(confidence)

from apps.cleancontact import utils
from apps.cleancontact.model import Model

from urllib.request import urlretrieve

import torch
import pandas as pd
from typing import Optional

class CLEANContact:

    def __init__(self):
        self.model = Model(512, 256, "cpu", torch.float32)
        model_path = "/home/zixu/clean-contact-website-backend/server/apps/cleancontact/ckpt.pth"
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        self.model.eval()

        self.emb_train = torch.load("/home/zixu/clean-contact-website-backend/server/apps/cleancontact/emb_train.pt", map_location="cpu", weights_only=False)
        self.id_ec_dict_train, self.ec_id_dict_train = utils.get_ec_id_dict("/home/zixu/clean-contact-website-backend/server/apps/cleancontact/train.csv")

        self.pvalue = 1e-5
        self.gmm = "/home/zixu/clean-contact-website-backend/server/apps/cleancontact/gmm_lst.pkl"

    def predict(self, seq: str, uniprot_id: str, pdb_file: Optional[str] = None):
        esm_x = utils.get_esm_x(seq)
        con_x = utils.get_con_x(pdb_file)
        
        emb_test = utils.model_embedding_test(esm_x, con_x, self.model)
        eval_dist = utils.get_dist_map_test(self.emb_train, emb_test, self.ec_id_dict_train, uniprot_id, "cpu", torch.float32)
        eval_df = pd.DataFrame.from_dict(eval_dist)

        rank_nk_ids, rank_nk_emb_train = utils.random_nk_model(self.id_ec_dict_train, self.ec_id_dict_train, self.emb_train, n=20, weighted=True)
        random_nk_dist_map = utils.get_random_nk_dist_map(self.emb_train, rank_nk_emb_train, self.ec_id_dict_train, rank_nk_ids, "cpu", torch.float32)

        pval_results, pval_confidence_results = utils.write_pvalue_choices(eval_df, random_nk_dist_map, self.pvalue, self.gmm)
        maxsep_results, maxsep_confidence_results = utils.write_max_sep_choices(eval_df, first_grad=True, use_max_grad=False, gmm=self.gmm)

        return pval_results, pval_confidence_results, maxsep_results, maxsep_confidence_results

    def run(self, uniprot_id: str, pdb_file: Optional[str] = None):

        try:
            assert uniprot_id is not None or pdb_file is not None, "Either uniprot_id or pdb_file must be provided"

            if pdb_file is None:
                try:
                    pdb_file, _ = urlretrieve(f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb")
                except Exception as e:
                    raise RuntimeError(f"Failed to download PDB file for {uniprot_id}: {e}")

            seq = utils.sequence_from_pdb(pdb_file)
            pval_results, pval_confidence_results, maxsep_results, maxsep_confidence_results = self.predict(seq, uniprot_id, pdb_file)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        return {
            "seq": seq,
            "pval_results": pval_results,
            "pval_confidence_results": pval_confidence_results,
            "maxsep_results": maxsep_results,
            "maxsep_confidence_results": maxsep_confidence_results,
            "status": "success",
            "message": "Predictions generated successfully"
        }

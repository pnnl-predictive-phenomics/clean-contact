import os
from Bio import PDB

def convert_cif_to_pdb(cif_folder, pdb_folder):
    # parser for .cif files
    cif_parser = PDB.MMCIFParser()

    # writer for .pdb files
    pdb_io = PDB.PDBIO()

    if not os.path.exists(pdb_folder):
        os.makedirs(pdb_folder)

    # iterate through .cif files in the input folder
    for file_name in os.listdir(cif_folder):
        print("file_name:", file_name)
        if file_name.endswith(".cif"):
            # construct the full file path
            cif_file_path = os.path.join(cif_folder, file_name)
            pdb_file_name = file_name.replace(".cif", ".pdb")
            pdb_file_path = os.path.join(pdb_folder, pdb_file_name)
            
            # parse the .cif file
            structure = cif_parser.get_structure(pdb_file_name, cif_file_path)
            
            # save the structure as a .pdb file
            pdb_io.set_structure(structure)
            pdb_io.save(pdb_file_path)
            print(f"Converted {file_name} to {pdb_file_name}")

if __name__ == "__main__":
    cif_folder = "cif_files"
    pdb_folder = "my_pdb"
    convert_cif_to_pdb(cif_folder, pdb_folder)

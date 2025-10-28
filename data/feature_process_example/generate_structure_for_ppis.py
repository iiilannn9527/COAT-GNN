import os
import torch
import esm
import numpy as np
from Bio import pairwise2
from pymol import cmd
import joblib
import pandas as pd
from Bio.PDB import *
from functools import reduce
def read_seq_file(seq_file):
    id_list = []
    seq_list = []
    lab_list = []
    with open(seq_file) as f:
        contents = f.readlines()
    for num, i in enumerate(contents):
        if i.startswith('>'):
            buffer = i.strip().replace('>', '')
            id_list.append(buffer)
            seq_list.append(contents[num+1].strip())
            lab_list.append(contents[num+2].strip())
    return id_list, seq_list, lab_list
def generate_seq_file_structure(id_list, seq_list, device, structure_dir):
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    model.set_chunk_size(1)
    for ix, (id, seq) in enumerate(zip(id_list, seq_list)):
        if not os.path.exists(f'{structure_dir}/{id}.pdb'):
            try:
                with torch.no_grad():
                    output = model.infer_pdb(seq)
                with open(f'{structure_dir}/{id}.pdb', "w") as f:
                    f.write(output)
                print(id, 'finish infer')
            except Exception as e:
                print(id, len(seq), e)

def generate_seq_file(id_list, seq_list, sequence_dir):
    for ix, (id, seq) in enumerate(zip(id_list, seq_list)):
        with open(f"{sequence_dir}/{id}.fasta", "w+") as f:
            f.write(f">{id}\n{seq}\n")

if __name__ == "__main__":
    '''
    define input file and output directory
    '''
    sequence_file = "example.txt"
    file_title = sequence_file.split(".")[0]
    structure_dir = f"structure_{file_title}"
    
    id_list, seq_list, lab_list = read_seq_file(sequence_file)
    #
    
    if not os.path.exists(structure_dir):
        os.makedirs(structure_dir, exist_ok=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        generate_seq_file_structure(id_list, seq_list, device, structure_dir)






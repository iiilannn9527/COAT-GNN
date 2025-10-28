'''
generating features
'''
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
# def predict_structure(model, sequence, out_name):
#     with torch.no_grad():
#         output = model.infer_pdb(sequence)
#     with open(out_name, "w") as f:
#         f.write(output)
def generate_seq_file(id_list, seq_list, sequence_dir):
    for ix, (id, seq) in enumerate(zip(id_list, seq_list)):
        with open(f"{sequence_dir}/{id}.fasta", "w+") as f:
            f.write(f">{id}\n{seq}\n")
def get_pssm(src_path, out_path):
    all_lists = os.listdir(src_path)
    file_lists_fas = []
    for file in all_lists:
        if file.endswith(".fasta"):
            file_name = f"{file.split('.')[0]}_u90.pssm"
            out_file = os.path.join(out_path, file_name)
            if not os.path.exists(out_file):
                file_lists_fas.append(file)
    for file in file_lists_fas:
        file_name = f"{file.split('.')[0]}_u90.pssm"
        src_file = os.path.join(src_path, file)
        out_file = os.path.join(out_path,file_name)
        if not os.path.exists(out_file):
            command = f"/public/home/rftang/software/blast+/bin/psiblast -query {src_file} -db /public/home/rftang/software/blast+/bin/uniref90.fasta -num_iterations 3 -evalue 0.001 -out_ascii_pssm {out_file}"
        #command = f'/public/home/rftang/software/ncbi-blast-2.15.0+/bin/psiblast -query {src_file} -db db_path -num_iterations 3 -evalue 0.001 -out_ascii_pssm {out_file}'
            os.system(command)
def get_hhm(src_path, out_path):
    all_lists = os.listdir(src_path)
    file_lists = []
    for file in all_lists:
        if file.endswith(".fasta"):
            file_lists.append(file)
    for file in file_lists:
        file_name = f"{file.split('.')[0]}.hhm"
        src_file = os.path.join(src_path, file)
        out_file = os.path.join(out_path,file_name)
        if not os.path.exists(out_file):
            command = f"hhblits -i {src_file} -ohhm {out_file} -n 2 -d /public/home/rftang/software/hh-suite/databases/uniclust30_2018_08/uniclust30_2018_08"
            os.system(command)
def process_pssm(file_name, file, ref_seq, out_path):
    with open(file_name, "r") as f:
        lines = f.readlines()
    p = 1 
    while len(lines[p].strip()) != 0:
        p += 1
    seq = ""
    pssm_list = []
    for i in range(3, p):
        res_list = lines[i].strip().split()
        seq += res_list[1]
        pssm_list.append(res_list[2:22])
    if seq == ref_seq:
        pssm_array = np.array(pssm_list).astype(float)
        np.save(f"{out_path}/{file}_pro_pssm.npy", pssm_array)
    else:
        raise f"seq of {file_name} is not equal"
def generate_PSSM_process(id_list, seq_list, src_path, out_path):
    for id_set, seq_set in zip(id_list, seq_list):
        src_file = f"{src_path}/{id_set}_u90.pssm"
        if os.path.exists(src_file):
            try:
                process_pssm(src_file, id_set, seq_set, out_path)
            except Exception as e:
                print("PSSM process ERRO!", id_set, e)
def process_hhm(src_file, id_set, ref_seq, out_path):
    with open(src_file, "r") as f:
        lines = f.readlines()
    
    for id_line, content in enumerate(lines):
        if content.strip() == "#":
            start = id_line+5
        if content.strip() == "//":
            end = id_line-1
    seq = ""
    hhm_list = []
    for i in range(start, end, 3):
        res_list = lines[i].strip().split()
        seq += res_list[0]
        hhm_set = res_list[2:22]
        for id_hhm, hhm in enumerate(hhm_set):
            if hhm == "*":
                hhm_set[id_hhm] = 0
            else:
                hhm_set[id_hhm] = float(hhm)
        hhm_list.append(hhm_set)
    if seq == ref_seq:
        hhm_array = np.array(hhm_list).astype(float)
        np.save(f"{out_path}/{id_set}_pro_hhm.npy", hhm_array)
    else:
        raise f"seq of {id_set} is not equal"
def generate_HHM_process(id_list, seq_list, src_path, out_path):
    for id_set, seq_set in zip(id_list, seq_list):
        src_file = f"{src_path}/{id_set}.hhm"
        if os.path.exists(src_file):
            process_hhm(src_file, id_set, seq_set, out_path)
def normalize(src_file, out_file, gloabl_max, global_min):
    # global gloabl_max, global_min
    data = np.load(src_file)
    new_data = (data - global_min) / (gloabl_max-global_min)
    np.save(out_file, new_data)

def normalize_hhm_pssm(id_list, seq_list, src_path, out_path, type_file):
    gloabl_max = -np.inf
    global_min = np.inf
    for id_set, seq_set in zip(id_list, seq_list):
        if len(seq_set) <= 5000:
            read_file = f"{src_path}/{id_set}_pro_{type_file}.npy"
            pssm_data = np.load(read_file)
            local_max = np.amax(pssm_data)
            local_min = np.amin(pssm_data)
            if local_max > gloabl_max:
                gloabl_max = local_max
            if local_min < global_min:
                global_min = local_min
    print('global_max', gloabl_max, 'global_min', global_min)
    for id_set, seq_set in zip(id_list, seq_list):
        if len(seq_set) <= 5000:
            src_file = f"{src_path}/{id_set}_pro_{type_file}.npy"
            if os.path.exists(src_file):
                out_file = os.path.join(out_path, f"{id_set}_normal_{type_file}.npy")
                normalize(src_file, out_file, gloabl_max, global_min)
            else:
                print(f"not exist {src_file}")



def get_dssp(src_path, out_path):
    file_list = os.listdir(src_path)
    for file in file_list:
        if file.endswith(".pdb"):
            file_name = f"{file.split('.')[0]}.dssp"
            src_file = os.path.join(src_path, file)
            out_file = os.path.join(out_path,file_name)
            if not os.path.exists(out_file):
                command = f"/public/home/rftang/software/miniconda3/envs/db_process/bin/mkdssp -i {src_file} -o {out_file}"
                os.system(command)

def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
        SS_vec[SS_type.find(SS)] = 1
        PHI = float(lines[i][103:109].strip())
        PSI = float(lines[i][109:115].strip())
        ACC = float(lines[i][34:38].strip())
        ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

    return seq, dssp_feature


def match_dssp(seq, dssp, ref_seq):
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    SS_vec = np.zeros(9) # The last dim represent "Unknown" for missing residues
    SS_vec[-1] = 1
    padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp


def transform_dssp(dssp_feature):
    dssp_feature = np.array(dssp_feature)
    angle = dssp_feature[:,0:2]
    ASA_SS = dssp_feature[:,2:]

    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

    return dssp_feature
def generate_dssp_process(id_list, seq_list, src_path, out_path):
    for id_set, seq_set in zip(id_list, seq_list):
        if os.path.exists(f"{src_path}/{id_set}.dssp"):
            if not os.path.exists(f"{out_path}/{id_set}.npy"):
                dssp_seq, dssp_matrix = process_dssp(f"{src_path}/{id_set}.dssp")
                if dssp_seq != seq_set:
                    dssp_matrix = match_dssp(dssp_seq, dssp_matrix, seq_set)
                np.save(f"{out_path}/{id_set}", transform_dssp(dssp_matrix))
def calculate_sasa(esmfold_path, file_name, query_path, selection="all"):
    cmd.load(f'{esmfold_path}/{file_name}.pdb')
    cmd.remove('hydro')
    cmd.remove('name OXT')
    cmd.set('dot_solvent', 1)
    cmd.set('dot_density', 4)
    # Run the get_area command which calculates the SASA and stores the result in stored.residues
    cmd.get_area(selection, load_b=1)
    cmd.save(f"{query_path}/{file_name}_sasa.pdb",'all')
    cmd.delete("all")
def def_atom_features():
    A = {'N':[0,0,0,1,0], 'CA':[0,0,0,1,0], 'C':[0,0,0,0,0], 'O':[0,0,0,0,0], 'CB':[0,0,0,3,0]}
    V = {'N':[0,0,0,1,0], 'CA':[0,0,0,1,0], 'C':[0,0,0,0,0], 'O':[0,0,0,0,0], 'CB':[0,0,0,1,0], 'CG1':[0,0,0,3,0], 'CG2':[0,0,0,3,0]}
    F = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0],'CB':[0,0,0,2,0],
         'CG':[1,1,0,0,1], 'CD1':[1,1,0,1,1], 'CD2':[1,1,0,1,1], 'CE1':[1,1,0,1,1], 'CE2':[1,1,0,1,1], 'CZ':[1,1,0,1,1] }
    P = {'N': [0,0,0, 0, 1], 'CA': [0,0,0, 1, 1], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0],'CB':[0,0,0,2,1], 'CG':[0,0,0,2,1], 'CD':[0,0,0,2,1]}
    L = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0], 'CG':[0,0,0,1,0], 'CD1':[0,0,0,3,0], 'CD2':[0,0,0,3,0]}
    I = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,1,0], 'CG1':[0,0,0,2,0], 'CG2':[0,0,0,3,0], 'CD1':[0,0,0,3,0]}
    R = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0],
         'CG':[0,0,0,2,0], 'CD':[0,0,0,2,0], 'NE':[0,0,0,1,0], 'CZ':[0,0,1,0,0], 'NH1':[0,0,0,2,0], 'NH2':[0,0,0,2,0] }
    D = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0], 'CG':[0,0,-1,0,0], 'OD1':[0,0,-1,0,0], 'OD2':[0,0,-1,0,0]}
    E = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0], 'CG':[0,0,0,2,0], 'CD':[0,0,-1,0,0], 'OE1':[0,0,-1,0,0], 'OE2':[0,0,-1,0,0]}
    S = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0], 'OG':[0,0,0,1,0]}
    T = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,1,0], 'OG1':[0,0,0,1,0], 'CG2':[0,0,0,3,0]}
    C = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0], 'SG':[0,0,-1,1,0]}
    N = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0], 'CG':[0,0,0,0,0], 'OD1':[0,0,0,0,0], 'ND2':[0,0,0,2,0]}
    Q = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0], 'CG':[0,0,0,2,0], 'CD':[0,0,0,0,0], 'OE1':[0,0,0,0,0], 'NE2':[0,0,0,2,0]}
    H = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0],
         'CG':[0,0,0,0,1], 'ND1':[0,0,-1,1,1], 'CD2':[0,0,0,1,1], 'CE1':[0,0,0,1,1], 'NE2':[0,0,-1,1,1]}
    K = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0], 'CG':[0,0,0,2,0], 'CD':[0,0,0,2,0], 'CE':[0,0,0,2,0], 'NZ':[0,0,0,3,1]}
    Y = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0],
         'CG':[1,1,0,0,1], 'CD1':[1,1,0,1,1], 'CD2':[1,1,0,1,1], 'CE1':[1,1,0,1,1], 'CE2':[1,1,0,1,1], 'CZ':[1,1,0,0,1], 'OH':[0,0,-1,1,0]}
    M = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0], 'CG':[0,0,0,2,0], 'SD':[0,0,0,0,0], 'CE':[0,0,0,3,0]}
    W = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 1, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0], 'CB':[0,0,0,2,0],
         'CG':[0,1,0,0,1], 'CD1':[0,1,0,1,1], 'CD2':[1,1,0,0,1], 'NE1':[0,1,0,1,1], 'CE2':[1,1,0,0,1], 'CE3':[1,1,0,1,1], 'CZ2':[1,1,0,1,1], 'CZ3':[1,1,0,1,1], 'CH2':[1,1,0,1,1]}
    G = {'N': [0,0,0, 1, 0], 'CA': [0,0,0, 2, 0], 'C': [0,0,0, 0, 0], 'O': [0,0,0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                   'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0],i_fea[1],i_fea[2]/2+0.5,i_fea[3]/3,i_fea[4]]

    return atom_features

def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict ={'GLY':'G','ALA':'A','VAL':'V','ILE':'I','LEU':'L','PHE':'F','PRO':'P','MET':'M','TRP':'W','CYS':'C',
               'SER':'S','THR':'T','ASN':'N','GLN':'Q','TYR':'Y','HIS':'H','ASP':'D','GLU':'E','LYS':'K','ARG':'R'}
    atom_count = -1
    res_count = -1
    pdb_file = open(file_path,'r')
    # pdb_res = pd.DataFrame(columns=['ID','atom','res','res_id','xyz','sasa_atom', 'mass', 'is_sidechain', 'is_benzene_c', 'atom_romatic', 
    #                                 'charge', 'num_H', 'ring'])
    pdb_res = pd.DataFrame()
    res_id_list = []
    chain_id_list = []
    before_res_pdb_id = None
    Relative_atomic_mass = {'H':1,'C':12,'O':16,'N':14,'S':32,'FE':56,'P':31,'BR':80,'F':19,'CO':59,'V':51,
                            'I':127,'CL':35.5,'CA':40,'B':10.8,'ZN':65.5,'MG':24.3,'NA':23,'HG':200.6,'MN':55,
                            'K':39.1,'AP':31,'AC':227,'AL':27,'W':183.9,'SE':79,'NI':58.7}

    index_line = 0
    while True:
        index_line += 1
        line = pdb_file.readline()
        #print(line)
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                print(f"has no relative atomic mass {atom_type}")
                continue
            atom_count+=1
            res_pdb_id = str(line[22:27].strip())
            chain_id = str(line[20:22].strip())
            if res_pdb_id != before_res_pdb_id:
                res_count +=1
            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ['N','CA','C','O','H']:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = res_dict[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                print(f'don not have feature atom of {atom} in {res}')
                atom_fea = [0.5, 0.5, 0.5,0.5,0.5]

            try:
                atom_sasa = float(line[60:66])
            except ValueError:
                raise f"do not have atom_sasa of {atom} in {res}"
                # atom_sasa = 0.5

            tmps = pd.Series(
                {'ID': atom_count, 'atom':line[12:16].strip(),'atom_type':atom_type, 'res': res, 'res_id': res_pdb_id,
                 'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                 'sasa_atom': atom_sasa,'mass':Relative_atomic_mass[atom_type],'is_sidechain':is_sidechain,
                 'is_benzene_c':atom_fea[0], 'atom_romatic':atom_fea[1],
                 'charge':atom_fea[2],'num_H':atom_fea[3],'ring':atom_fea[4], "chain_id":chain_id}).to_frame()
            if len(res_id_list) == 0:
                res_id_list.append(res_pdb_id)
                chain_id_list.append(chain_id)
            elif res_id_list[-1] != res_pdb_id:
                res_id_list.append(res_pdb_id)
                chain_id_list.append(chain_id)
            pdb_res = pd.concat([pdb_res, tmps.T], axis=0)
            # pdb_res = pdb_res.append(tmps, ignore_index=True)
        if line.startswith('END'):
            #print('exist TER')
            pdb_file.close()
            break
        # if index_line > 9999:
        #     print(f'the code in circle over 9999 in {file_path}')
            # break

    return pdb_res,res_id_list, chain_id_list

def PDBFeature(file_pdb_path, file_name, query_out_path): #

    #print('PDB_chain -> PDB_DF')
    pdb_path = f'{file_pdb_path}/{file_name}_sasa.pdb'
    pdb_DF, res_id_list, chain_id_list = get_pdb_DF(pdb_path)

    with open(query_out_path+'/{}.df'.format(file_name),'wb') as f:
        joblib.dump({'pdb_DF':pdb_DF,'res_id_list':res_id_list, "chain_id_list":chain_id_list},f)

    #print('Extract PDB_feature')
    #print(file_name)

    res_sidechain_centroid = []
    res_types = []
    for res_id, chain_id in zip(res_id_list, chain_id_list):
        res_type = pdb_DF[(pdb_DF['res_id'] == res_id) & (pdb_DF['chain_id'] == chain_id)]['res'].values[0]
        res_types.append(res_type)

        res_atom_df = pdb_DF[(pdb_DF['res_id'] == res_id) & (pdb_DF['chain_id'] == chain_id)]
        xyz = np.array(res_atom_df['xyz'].tolist())
        masses = np.array(res_atom_df['mass'].tolist()).reshape(-1,1)
        centroid = np.sum(masses*xyz,axis=0)/np.sum(masses)
        res_sidechain_atom_df = pdb_DF[(pdb_DF['res_id'] == res_id) & (pdb_DF['is_sidechain'] == 1) & (pdb_DF['chain_id'] == chain_id)]
        if len(res_sidechain_atom_df) == 0:
            res_sidechain_centroid.append(centroid)
        else:
            xyz = np.array(res_sidechain_atom_df['xyz'].tolist())
            masses = np.array(res_sidechain_atom_df['mass'].tolist()).reshape(-1, 1)
            sidechain_centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
            res_sidechain_centroid.append(sidechain_centroid)

    res_sidechain_centroid = np.array(res_sidechain_centroid)
    with open(query_out_path + '/'+file_name+'_psepos_SC.pkl', 'wb') as f:
        joblib.dump(res_sidechain_centroid, f)
    return
def PDBResidueFeature(query_out_path,file_name):

    atom_vander_dict = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.85,'H':1.2,'D':1.2,'SE':1.9,'P':1.8,'FE':2.23,'BR':1.95,
                        'F':1.47,'CO':2.23,'V':2.29,'I':1.98,'CL':1.75,'CA':2.81,'B':2.13,'ZN':2.29,'MG':1.73,'NA':2.27,
                        'HG':1.7,'MN':2.24,'K':2.75,'AC':3.08,'AL':2.51,'W':2.39,'NI':2.22}
    for key in atom_vander_dict.keys():
        atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)

    with open('{}/{}.df'.format(query_out_path,file_name), 'rb') as f:
        tmp = joblib.load(f)
    pdb_DF, res_id_list, chain_id_list = tmp['pdb_DF'], tmp['res_id_list'], tmp['chain_id_list']
    pdb_DF = pdb_DF[pdb_DF['atom_type']!='H']

    # atom features
    mass = np.array(pdb_DF['mass'].tolist()).reshape(-1, 1)
    mass = mass / 32
    sasa_atom = np.array(pdb_DF['sasa_atom'].tolist()).reshape(-1, 1)
    if (max(sasa_atom) - min(sasa_atom)) == 0:
        SASA_atom = np.zeros(sasa_atom.shape) + 0.5
    else:
        SASA_atom = (sasa_atom - min(sasa_atom)) / (max(sasa_atom) - min(sasa_atom))
    benzene_c = np.array(pdb_DF['is_benzene_c'].tolist()).reshape(-1, 1)
    atom_romatic = np.array(pdb_DF['atom_romatic'].tolist()).reshape(-1, 1)
    is_sidechain = np.array(pdb_DF['is_sidechain'].tolist()).reshape(-1, 1)
    charge = np.array(pdb_DF['charge'].tolist()).reshape(-1, 1)
    num_H = np.array(pdb_DF['num_H'].tolist()).reshape(-1, 1)
    ring = np.array(pdb_DF['ring'].tolist()).reshape(-1, 1)
    atom_type = pdb_DF['atom_type'].tolist()
    atom_vander = np.zeros((len(atom_type), 1))
    for i, type in enumerate(atom_type):
        try:
            atom_vander[i] = atom_vander_dict[type]
        except:
            atom_vander[i] = atom_vander_dict['C']

    atom_feas = [benzene_c, atom_romatic, SASA_atom, mass, is_sidechain, charge, num_H, ring, atom_vander]
    atom_feas = np.concatenate(atom_feas,axis=1)

    res_atom_feas = []
    atom_begin = 0
    for i, (res_id, chain_id) in enumerate(zip(res_id_list, chain_id_list)):
        res_atom_df = pdb_DF[(pdb_DF['res_id'] == res_id) & (pdb_DF['chain_id'] == chain_id)]
        atom_num = len(res_atom_df)
        res_atom_feas_i = atom_feas[atom_begin:atom_begin + atom_num]
        res_atom_feas_i = np.average(res_atom_feas_i, axis=0).reshape(1, -1)
        res_atom_feas.append(res_atom_feas_i)
        atom_begin += atom_num
    res_atom_feas = np.concatenate(res_atom_feas, axis=0)


    residue_feas = [res_atom_feas]
    residue_feas = np.concatenate(residue_feas, axis=1)
    with open('{}/{}.resfea'.format(query_out_path,file_name),'wb') as f:
        joblib.dump(residue_feas,f)
    return
def SAResidueFeature(query_out_path, file_name , SASA_out_path):
    with open('{}/{}.df'.format(query_out_path,file_name), 'rb') as f:
        tmp = joblib.load(f)
    pdb_DF, res_id_list, chain_id_list = tmp['pdb_DF'], tmp['res_id_list'], tmp['chain_id_list']
    pdb_DF = pdb_DF[pdb_DF['atom_type']!='H']
    SASA_atom = np.array(pdb_DF['sasa_atom'].tolist()).reshape(-1, 1)
    assert (np.max(SASA_atom) - np.min(SASA_atom)) > 0.
    #     SASA_atom = np.zeros(sasa_atom.shape) + 0.5
    # else:
    #     SASA_atom = (sasa_atom - min(sasa_atom)) / (max(sasa_atom) - min(sasa_atom))
    
    atom_feas = [SASA_atom]
    atom_feas = np.concatenate(atom_feas,axis=1)

    res_atom_sasa_sum = []
    res_atom_sasa_ave = []
    atom_begin = 0
    for i, (res_id, chain_id) in enumerate(zip(res_id_list, chain_id_list)):
        res_atom_df = pdb_DF[(pdb_DF['res_id'] == res_id) & (pdb_DF['chain_id'] == chain_id)]
        atom_num = len(res_atom_df)
        res_atom_feas_i = atom_feas[atom_begin:atom_begin + atom_num]
        res_atom_feas_ave = np.average(res_atom_feas_i, axis=0).reshape(1, -1)
        res_atom_feas_sum = np.sum(res_atom_feas_i, axis=0).reshape(1, -1)
        res_atom_sasa_sum.append(res_atom_feas_sum)
        res_atom_sasa_ave.append(res_atom_feas_ave)
        atom_begin += atom_num
    res_atom_sasa_sum = np.concatenate(res_atom_sasa_sum, axis=0)
    res_atom_sasa_ave = np.concatenate(res_atom_sasa_ave, axis=0)


    residue_feas = [res_atom_sasa_sum, res_atom_sasa_ave]
    residue_feas = np.concatenate(residue_feas, axis=1)
    with open('{}/{}_SASA.pkl'.format(SASA_out_path, file_name),'wb') as f:
        joblib.dump(residue_feas,f)
    return
def Feature(esmfold_path, query_path, query_out_path, SASA_out_path):
    file_lists = os.listdir(esmfold_path)
    # if len(os.listdir(query_path)) == 0:
    #     filename_lists = []
    filename_lists = []
    for file_set in file_lists:
        if file_set.endswith('pdb'):
            filename = file_set.split('.')[0]
            if not os.path.exists(f"{query_path}/{filename}_sasa.pdb"):
                calculate_sasa(esmfold_path, filename, query_path)
            filename_lists.append(filename)
    for filename in filename_lists:
        with open('{}/{}_sasa.pdb'.format(query_path, filename),'r') as f:
            text = f.readlines()
        residue_num = 0
        for line in text:
            if line.startswith('ATOM'):
                residue_type = line[17:20]
                if residue_type not in ["GLY","ALA","VAL","ILE","LEU","PHE","PRO","MET","TRP","CYS",
                                        "SER","THR","ASN","GLN","TYR","HIS","ASP","GLU","LYS","ARG"]:
                    print("ERROR: There are mutant residues in your structure!")
                    print(filename, residue_type)
                    raise ValueError

                residue_num += 1
        if residue_num == 0:
            print('ERROR: Your query chain id is not in the uploaded structure, please check the chain ID!')
            raise ValueError
        #print('2.extracting features...')
        PDBFeature(query_path, filename, query_out_path)
        PDBResidueFeature(query_out_path,filename)
        SAResidueFeature(query_out_path, filename , SASA_out_path)
def generate_atom_feature(structure_file, structure_sasa_dir, atom_dir, sasa_dir):   
    os.makedirs(structure_sasa_dir, exist_ok = True)
    os.makedirs(atom_dir, exist_ok = True)
    os.makedirs(sasa_dir, exist_ok = True)
    Feature(structure_file, structure_sasa_dir, atom_dir, sasa_dir)


def process_gly(resi, file_name):
    cmd.select("buffer", f"{file_name} and resi {resi} and hydro and neighbor name CA")
    gly_list = [atom.id for atom in cmd.get_model('buffer').atom]
    if len(gly_list) == 2:
        cmd.remove(f"{file_name} and id {gly_list[0]}")
    cmd.delete("buffer")
def pymol_rmh(src_file, file_name, out_file):
    cmd.load(src_file, file_name)
    # cmd.h_add("name CA")
    # cmd.h_add()
    cmd.h_add(f"(not ({file_name} and backbone)) and {file_name}")
    cmd.h_add(f"{file_name} and resn GLY and name CA")
    cmd.select('GLY_ID', f'{file_name} and resn GLY')
    gly_list1 = []
    for residue in cmd.get_model("GLY_ID").atom:
        id = residue.resi.strip("'")
        if id not in gly_list1:
            gly_list1.append(id)
    for i in gly_list1:
        process_gly(i, file_name)  
    cmd.save(out_file, file_name)
    cmd.delete(file_name)
def calc_dist(coord1, coord2):
    coord1, coord2 = np.array(coord1), np.array(coord2)
    return np.linalg.norm(coord1 - coord2)

def pesudom_gen(id, pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(id, pdb_path)
    residue_centers = []
    for chain in structure[0]:
        for residue in chain:
            side_chain_atoms = [atom for atom in residue.get_unpacked_list() if not atom.id in ['N', 'CA', 'C', 'O']]
            if side_chain_atoms:
                total = reduce(lambda x, y: x + y.coord, side_chain_atoms, 0)
                center = total/len(side_chain_atoms)
                # center = [sum(atom.coord)/len(side_chain_atoms) for atom in side_chain_atoms]
                residue_centers.append(center)

            else:
                print(f"sidechain of {id} cannot be none! {len(residue_centers)}")
                residue_centers = []
                break
                # raise ValueError(f"sidechain of {id} cannot be none! {len(residue_centers)}")
               
    if residue_centers:
        first_center = residue_centers[0]
        dis_result = []
        for i, center in enumerate(residue_centers):
            dist = calc_dist(first_center, center)
            # print('Distance from first residue to residue {}: {}'.format(i, dist))
            dis_result.append(dist)
        return dis_result
    else:
        return []
def generate_pseudom(src_path, npy_path):
    file_list = os.listdir(src_path)
    # erro_list_file = []
    erro_list = []
    for file in file_list:
        if file.endswith(".pdb"):
            file_name = file.split(".")[0]
            file_name3 = f"{file_name}.npy"
            src_file = os.path.join(src_path, file)
            npy_file = os.path.join(npy_path, file_name3)
            if not os.path.exists(npy_file):
                dis_result = pesudom_gen(file_name, src_file)
                if not dis_result:
                    erro_list.append(file_name)
                    continue
                dis_array = np.array(dis_result)
                
                np.save(npy_file, dis_array)

    erro_str = ','.join(erro_list)
    print(erro_str)
if __name__ == "__main__":
    '''
    define input file and output dict
    '''
    sequence_file = "example.txt"
    file_title = sequence_file.split(".")[0]
    structure_dir = f"structure_{file_title}"
    sequence_dir = f"fas_{file_title}"
    PSSM_out_dir = f"pssm_u90_{file_title}"
    HHM_out_dir = f"hhm_{file_title}"
    PSSM_process_dir = f"pssm_u90_process_{file_title}"
    HHM_process_dir = f"hhm_process_{file_title}"
    DSSP_out_dir = f"dssp_{file_title}"
    DSSP_process_dir = f"dssp_process_{file_title}"
    schordinger_structure_dir = f"structure_{file_title}_schordinger"
    pseudom_dir = f"pseudom_{file_title}"
    structure_sasa_dir = f"sasa_{file_title}"
    atom_dir = f"atomfea_{file_title}"
    sasa_dir = f"safea_{file_title}"
    PSSM_normal_dir = f"pssm_u90_normal_{file_title}"
    HHM_normal_dir = f"hhm_normal_{file_title}"

    id_list, seq_list, lab_list = read_seq_file(sequence_file)
    
    if not os.path.exists(structure_dir):
        os.makedirs(structure_dir, exist_ok=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        generate_seq_file_structure(id_list, seq_list, device, structure_dir)
    
    if not os.path.exists(sequence_dir):
        os.makedirs(sequence_dir, exist_ok=False)
        generate_seq_file(id_list, seq_list, sequence_dir)
    #PSSM,HHM
    if not os.path.exists(PSSM_out_dir):
        os.makedirs(PSSM_out_dir, exist_ok = False)
        get_pssm(sequence_dir, PSSM_out_dir)
    
    if not os.path.exists(HHM_out_dir):
        os.makedirs(HHM_out_dir, exist_ok = False)
        get_hhm(sequence_dir, HHM_out_dir)
    
    if not os.path.exists(PSSM_process_dir):
        os.makedirs(PSSM_process_dir, exist_ok = False)
        generate_PSSM_process(id_list, seq_list, PSSM_out_dir, PSSM_process_dir)
    if not os.path.exists(HHM_process_dir):
        os.makedirs(HHM_process_dir, exist_ok = False)
        generate_HHM_process(id_list, seq_list, HHM_out_dir, HHM_process_dir)
    #DSSP
    if not os.path.exists(DSSP_out_dir):
        os.makedirs(DSSP_out_dir, exist_ok = False)
        get_dssp(structure_dir, DSSP_out_dir)
    
    os.makedirs(DSSP_process_dir, exist_ok = True)
    generate_dssp_process(id_list, seq_list, DSSP_out_dir, DSSP_process_dir)
    #atom feature
    if not os.path.exists(structure_sasa_dir):
        os.makedirs(structure_sasa_dir, exist_ok = True)
        os.makedirs(atom_dir, exist_ok = True)
        os.makedirs(sasa_dir, exist_ok = True)
        generate_atom_feature(structure_dir, structure_sasa_dir, atom_dir, sasa_dir)
    
    if not os.path.exists(PSSM_normal_dir):
        os.makedirs(PSSM_normal_dir, exist_ok = True)
        normalize_hhm_pssm(id_list, seq_list, PSSM_process_dir, PSSM_normal_dir, "pssm")
    if not os.path.exists(HHM_normal_dir):
        os.makedirs(HHM_normal_dir, exist_ok = True)   
        normalize_hhm_pssm(id_list, seq_list, HHM_process_dir, HHM_normal_dir, "hhm")
    #pseudom
    os.makedirs(pseudom_dir, exist_ok = True)
    generate_pseudom(schordinger_structure_dir, pseudom_dir)






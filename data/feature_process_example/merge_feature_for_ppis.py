import os
import pickle
import numpy as np
import joblib
def calc_dist(coord1, coord2):
    coord1, coord2 = np.array(coord1), np.array(coord2)
    return np.linalg.norm(coord1 - coord2)
def get_dis(residue_centers):
    first_center = residue_centers[0]
    dis_result = []
    for i, center in enumerate(residue_centers):
        dist = calc_dist(first_center, center)
        # print('Distance from first residue to residue {}: {}'.format(i, dist))
        dis_result.append(dist)
    return dis_result
def gen_last_pssm(id_lst, ref_pssm_dir, id_seq_dict):
    out_dict = {}
    for id_set in id_lst:
        buffer = np.load(f"{ref_pssm_dir}/{id_set}_normal_pssm.npy")
        if buffer.shape[0] == len(id_seq_dict[id_set]):
            out_dict[id_set] = buffer
        else:
            print("pssm erro match", id_set)
    print("pssm shape:",buffer.shape)
    return out_dict
def gen_last_hhm(id_lst, ref_hhm_dir, id_seq_dict):
    out_dict = {}
    for id_set in id_lst:
        buffer = np.load(f"{ref_hhm_dir}/{id_set}_normal_hhm.npy")
        if buffer.shape[0] == len(id_seq_dict[id_set]):
            out_dict[id_set] = buffer
        else:
            print("hhm erro match", id_set)
    print("hhm shape:",buffer.shape)
    return out_dict
def gen_last_dssp(id_lst, ref_dssp_dir, id_seq_dict):
    out_dict = {}
    for id_set in id_lst:
        buffer = np.load(f"{ref_dssp_dir}/{id_set}.npy")
        if buffer.shape[0] == len(id_seq_dict[id_set]):
            out_dict[id_set] = buffer
        else:
            print("dssp erro match", id_set)
    print("dssp shape:",buffer.shape)
    return out_dict
def gen_last_resfea(id_lst, ref_resfea_dir, id_seq_dict):
    out_dict = {}
    for id_set in id_lst:
        buffer = joblib.load(f"{ref_resfea_dir}/{id_set}.resfea")
        if buffer.shape[0] == len(id_seq_dict[id_set]):
            out_dict[id_set] = buffer
        else:
            print("resfea erro match", id_set)
    print("shape of atom feature (9 dim):",buffer.shape)
    return out_dict
def gen_last_sasa(id_lst, ref_sasa_dir, id_seq_dict):
    out_dict = {}
    for id_set in id_lst:
        buffer = joblib.load(f"{ref_sasa_dir}/{id_set}_SASA.pkl")
        if buffer.shape[0] == len(id_seq_dict[id_set]):
            out_dict[id_set] = buffer
        else:
            print("sasa erro match", id_set)
    print("shape of atom feature for sasa:",buffer.shape)
    return out_dict
def gen_last_SC_centroid(id_lst, ref_SC_dir, id_seq_dict):
    out_dict = {}
    for id_set in id_lst:
        read_file = f"{ref_SC_dir}/{id_set}_psepos_SC.pkl"
        if os.path.exists(read_file):
            buffer = joblib.load(f"{ref_SC_dir}/{id_set}_psepos_SC.pkl")
            if buffer.shape[0] == len(id_seq_dict[id_set]):
                out_dict[id_set] = buffer
            else:
                print("psepos_SC erro match", id_set)
    print("shape of SC centroid coordinates", buffer.shape)
    return out_dict
def gen_SC_dismap_pseudom(SC_centroid_dict):
    out_dict_pseudom = {}
    out_dict_dismap_SC = {}
    for key, value in SC_centroid_dict.items():
        dis_result = get_dis(value)
        out_dict_pseudom[key] = np.array(dis_result)
        num_residues = value.shape[0]
        dist_matrix = np.zeros((num_residues, num_residues))
        for i in range(num_residues):
            for j in range(i+1, num_residues):
                distance = calc_dist(value[i], value[j])
                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance
        out_dict_dismap_SC[key] = dist_matrix.astype(np.float32)
    print("shape of SC centroid dismap", dist_matrix.shape, "shape of SC_centroid pseudom", np.array(dis_result).shape)
    return out_dict_pseudom, out_dict_dismap_SC

def gen_geometric_pseudom(id_lst, geometric_dir,id_seq_dict):
    out_dict = {}
    for id_set in id_lst:
        buffer = np.load(f"{geometric_dir}/modified_{id_set}.npy")
        if buffer.shape[0] == len(id_seq_dict[id_set]):
            out_dict[id_set] = buffer
        else:
            print("geometric_centroid erro match", id_set)
    print("the shape of geometric center coordinates:\n",buffer.shape)
    return out_dict

# def gen_SC_geometric_pseudom(SC_geometric_dict):
#     out_dict_pseudom = {}
#     for key, value in SC_geometric_dict.items():
#         dis_result = get_dis(value)
#         out_dict_pseudom[key] = np.array(dis_result)
#     print("the shape of geometric center pseudom", np.array(dis_result).shape)
#     return out_dict_pseudom
if __name__ == "__main__":
    # sequence_file = "example.txt"
    # file_title = sequence_file.split(".")[0]
    # structure_dir = f"structure_{file_title}"
    # sequence_dir = f"fas_{file_title}"
    # PSSM_out_dir = f"pssm_u90_{file_title}"
    # HHM_out_dir = f"hhm_{file_title}"
    # PSSM_process_dir = f"pssm_u90_process_{file_title}"
    # HHM_process_dir = f"hhm_process_{file_title}"
    # DSSP_out_dir = f"dssp_{file_title}"
    # DSSP_process_dir = f"dssp_process_{file_title}"
    # schordinger_structure_dir = f"structure_{file_title}_schordinger"
    # pseudom_dir = f"pseudom_{file_title}"
    # structure_sasa_dir = f"sasa_{file_title}"
    # atom_dir = f"atomfea_{file_title}"
    # sasa_dir = f"safea_{file_title}"
    # PSSM_normal_dir = f"pssm_u90_normal_{file_title}"
    # HHM_normal_dir = f"hhm_normal_{file_title}"

    file_txt = "example.txt"
    standard_prefix = file_txt.split(".")[0]
    pssm_get_dir = f"pssm_u90_normal_{standard_prefix}"
    hhm_get_dir = f"hhm_normal_{standard_prefix}"
    dssp_get_dir = f"dssp_process_{standard_prefix}"
    atom_fea_get_dir = f"atomfea_{standard_prefix}"
    sc_get_dir = f"atomfea_{standard_prefix}"
    sa_get_dir = f"safea_{standard_prefix}"
    pseudom_get_dir = f"pseudom_{standard_prefix}"

    id_seq_dict = {}
    id_lab_dict = {}
    with open(file_txt, "r") as f:
        contents = f.readlines()
    for num, i in enumerate(contents):
        if i.startswith('>'):
            buffer = i.strip().replace('>', '')
            id_seq_dict[buffer] = contents[num+1].strip()
            id_lab_dict[buffer] = contents[num+2].strip()
    
    SC_centroid_dict = gen_last_SC_centroid(list(id_seq_dict.keys()), sc_get_dir, id_seq_dict)
    with open(f"{standard_prefix}_SC_coordinate.pkl", "wb") as f:
        pickle.dump(SC_centroid_dict, f)
    pdb_match_lst = list(SC_centroid_dict.keys())
    out_dict_pseudom, out_dict_dismap_SC = gen_SC_dismap_pseudom(SC_centroid_dict)
    # with open(f"{standard_prefix}_SC_pseudom.pkl", "wb") as f:
    #     pickle.dump(out_dict_pseudom, f)
    with open(f"{standard_prefix}_SC_dismap.pkl", "wb") as f:
        pickle.dump(out_dict_dismap_SC, f)
    
    out_dict_pseudom = gen_geometric_pseudom(pdb_match_lst, pseudom_get_dir, id_seq_dict)
    with open(f"{standard_prefix}_SC_geometric_pseudom.pkl", "wb") as f:
        pickle.dump(out_dict_pseudom, f)
    
    with open(f"{standard_prefix}_process_seq_lab.txt", "w+") as f1:
        for id_set in pdb_match_lst:
            f1.write(f">{id_set}\n{id_seq_dict[id_set]}\n{id_lab_dict[id_set]}\n")
    
    # with open(f"{standard_prefix}_{txtname}_ca_distancemap.pkl", "wb") as f:
    #     ca_dict = {}
    #     for id_set in pdb_match_lst:
    #         ca_dict[id_set] = pdb_match_data[id_set]
    #     pickle.dump(ca_dict, f)

    # with open(f"ca_dismap_{txtname}_retain_pdb3_fixer.pkl", "rb") as f:
    #     pdb_match_data = pickle.load(f)

    out_dict = gen_last_pssm(pdb_match_lst, pssm_get_dir, id_seq_dict)
    with open(f"{standard_prefix}_pssm.pkl", "wb") as f:
        pickle.dump(out_dict, f)
    out_dict = gen_last_hhm(pdb_match_lst, hhm_get_dir, id_seq_dict)
    with open(f"{standard_prefix}_hhm.pkl", "wb") as f:
        pickle.dump(out_dict, f)
    out_dict = gen_last_dssp(pdb_match_lst, dssp_get_dir, id_seq_dict)
    with open(f"{standard_prefix}_dssp.pkl", "wb") as f:
        pickle.dump(out_dict, f)
    
    out_dict = gen_last_resfea(pdb_match_lst, sc_get_dir, id_seq_dict)
    with open(f"{standard_prefix}_resfea.pkl", "wb") as f:
        pickle.dump(out_dict, f)
    out_dict = gen_last_sasa(pdb_match_lst, sa_get_dir, id_seq_dict)
    with open(f"{standard_prefix}_SASA.pkl", "wb") as f:
        pickle.dump(out_dict, f)

    
    # with open(f"ca_vector_{txtname}_retain_pdb3_fixer.pkl", "rb") as f:
    #     ca_coordinate_data = pickle.load(f)
    # with open(f"{standard_prefix}_{txtname}_ca_coordinate.pkl", "wb") as f:
    #     ca_coordinate = {}
    #     for id_set in pdb_match_lst:
    #         ca_coordinate[id_set] = ca_coordinate_data[id_set]
    #     pickle.dump(ca_coordinate, f)

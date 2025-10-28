import os

from tqdm import tqdm
import random


def read_data(file_path, min_seq_len=0, max_seq_len=500, distance_dic=None):
    assert max_seq_len > min_seq_len

    files = open(file_path, mode='r')

    try:
        files_lines = files.readlines()
    finally:
        files.close()

    name = files_lines[0::3]
    amoni_acids = files_lines[1::3]
    labels = files_lines[2::3]
    #
    name = [i[1:-1] for i in name]
    amoni_acids = [i[:-1] for i in amoni_acids]
    labels = [i[:-1] for i in labels]

    assert len(name) == len(amoni_acids) == len(labels)

    labels_pro = []
    #
    for i in range(len(labels)):
        labels_pro.append([int(label) for label in labels[i]])

    name_500 = []
    amoni_acids_500 = []
    labels_500 = []
    max_len = 0

    for i, label in enumerate(labels_pro):
        max_len = max(max_len, len(label))
        if max_seq_len >= len(label) > min_seq_len:
            if distance_dic is None:
                if name[i] not in ['2j3rA', '6cdi_D', '6cuf_D', '6cuf_C', '6cdi_c']:
                    # and name[i] != '1jmoH' and name[i] != '3fqdA' and name[i] != '1jmoL' \
                    # and name[i] != '3cbjA':
                    name_500.append(name[i])
                    amoni_acids_500.append(amoni_acids[i])
                    labels_500.append(label)

            elif name[i] in distance_dic:
                if name[i] not in ['2j3rA', '6cdi_D', '6cuf_D', '6cuf_C', '6cdi_c']:
                    # and name[i] != '1jmoH' and name[i] != '3fqdA' and name[i] != '1jmoL'\
                    # and name[i] != '3cbjA' :
                    name_500.append(name[i])
                    amoni_acids_500.append(amoni_acids[i])
                    labels_500.append(label)

                    assert len(amoni_acids[i]) == len(label)

    # max_num = 0
    # count_1000 = 0
    # count_500 = 0
    # count_0 = 0
    # for label in tqdm(labels_pro):
    #     label_len = len(label)
    #     if label_len > 1000:
    #         count_1000 += 1
    #     elif label_len > 500:
    #         count_500 += 1
    #     else:
    #         count_0 += 1
    #     max_num = max(max_num, len(label))
    # print(max_num)
    # print(count_0, count_500, count_1000)

    tokens = [(name_500[i], amoni_acids_500[i]) for i in range(len(amoni_acids_500))]
    # print(max_len)

    return tokens, labels_500


# def train_valid_set(tokens, labels, proteins_id):
#     data_index = [i for i in range(tokens.size(0))]
#     random.shuffle(data_index)
#     train_index = data_index[:int(len(data_index) * 0.9)]
#     valid_index = data_index[int(len(data_index) * 0.9):]
#
#     train_tokens = tokens[train_index]
#     train_labels = [labels[index] for index in train_index]
#     train_proteins_id = [proteins_id[index] for index in train_index]
#
#     valid_tokens = tokens[valid_index]
#     valid_labels = [labels[index] for index in valid_index]
#     valid_proteins_id = [proteins_id[index] for index in valid_index]
#
#     return (train_tokens, train_labels, train_proteins_id), (valid_tokens, valid_labels, valid_proteins_id)


# def build_dataset(train_file_path, residue_psepos_dict, batch_converter, dataset_distance_dic, dataset_graphs, alphabet,
#                   xyz_train_path, train_sasa_path, train_init_cluster_path):
#     train_set, labels = read_data(train_file_path, 0, 1000, residue_psepos_dict)
#     proteins_id, strs, tokens = batch_converter(train_set)
#     train_data, valid_data = train_valid_set(tokens, labels, proteins_id)
#     dataset = Dataset_Cluster_335(train_data[0], train_data[1], train_data[2], dataset_distance_dic,
#                                   dataset_graphs, alphabet,
#                                   path='data/database-2/Feature/psepos/Train335_psepos_SC.pkl',
#                                   xyz_path=xyz_train_path, contact_area_path=train_sasa_path,
#                                   reference_radius=20.,
#                                   init_cluster=train_init_cluster_path,
#                                   train_mode=True)
#     valid_dataset = Dataset_Cluster_335(valid_data[0], valid_data[1], valid_data[2],
#                                         dataset_distance_dic,
#                                         dataset_graphs, alphabet,
#                                         path='data/database-2/Feature/psepos/Train335_psepos_SC.pkl',
#                                         xyz_path=xyz_train_path, contact_area_path=train_sasa_path,
#                                         reference_radius=20.,
#                                         init_cluster=train_init_cluster_path,
#                                         train_mode=False)
#
#     return dataset, valid_dataset


# token, labels = read_data('./data/database-1/Train_9982_Pid_Pseq_label.txt', 400, 500)

def train_dev(file_path, save_path, rate=0.8):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files = open(file_path, mode='r')

    try:
        files_lines = files.readlines()
    finally:
        files.close()

    name = files_lines[0::3]
    amoni_acids = files_lines[1::3]
    labels = files_lines[2::3]

    samples_num = len(name)
    folds_num = int(1 / (1 - rate))
    one_fold_samples = samples_num // folds_num

    indexes = [i for i in range(samples_num)]
    random.shuffle(indexes)

    for f_n in range(folds_num):
        valid_index = indexes[f_n * one_fold_samples: (f_n + 1) * one_fold_samples]
        train_index = []
        for i in indexes:
            if i not in valid_index:
                train_index.append(i)
        assert len(valid_index) + len(train_index) == samples_num

        train_name = [name[key] for key in train_index]
        train_aa_seq = [amoni_acids[key] for key in train_index]
        train_labels = [labels[key] for key in train_index]
        test_name = [name[key] for key in valid_index]
        test_aa_seq = [amoni_acids[key] for key in valid_index]
        test_labels = [labels[key] for key in valid_index]

        file = open(save_path + f'{f_n}-train.txt', 'w')
        for i in range(len(train_index)):
            file.write(str(train_name[i]))
            file.write(str(train_aa_seq[i]))
            file.write(str(train_labels[i]))
        file.close()

        file = open(save_path + f'{f_n}-valid.txt', 'w')
        for i in range(len(valid_index)):
            file.write(str(test_name[i]))
            file.write(str(test_aa_seq[i]))
            file.write(str(test_labels[i]))
        file.close()


# train_dev('data/database-2/Train_335.fa', 0.8)
# for seed in [2, 12, 22, 32, 42]:
#     random.seed(seed)
#     train_dev('data/database-2/Train_335.fa', f'data/database-2/{seed}_folds/', 0.8)

#coding:gbk
import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
from functools import reduce
from numpy import *




def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True

"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

#folder = "新数据/"
folder = ""

def load_drug_list():
    filename = folder + "Drug_listFri Feb 10 05_41_20 2023.csv"
    csvfile = open(filename, "r")
    reader = csv.reader(csvfile)
    next(reader, None)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs

def write_drug_cid():
    drugs = load_drug_list()
    drug_id = []
    datas = []
    outputfile = open(folder + 'pychem_cid.csv', 'wb')
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid
        print(drug, cid)
        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    outputfile = open(folder + "unknow_drug_by_pychem.csv", 'wb')
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)

def cid_from_other_source():
    """
    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(folder + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    next(reader)
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict: 
            cid_dict[name] = str(cid)

    unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k:v for k,v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict

def load_cid_dict():
    reader = csv.reader(open(folder + "pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict

def download_smiles():
    cids_dict = load_cid_dict()
    cids = [v for k,v in cids_dict.iteritems()]
    inv_cids_dict = {v:k for k,v in cids_dict.iteritems()}
    download('CSV', folder + 'drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES', overwrite=True)
    f = open(folder + 'drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + next(reader)
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(folder + "drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()

"""
The following code will convert the SMILES format into onehot format
"""
def string2smiles_list(string):
    char_list = []
    i = 1
    while i < len(string):
        c = string[i]
        if c.islower():
            char_list.append(string[i-1:i+1])
            i += 1
        else:
            char_list.append(string[i-1])
        i += 1
    if not string[-1].islower():
        char_list.append(string[-1])
    return char_list

def string2smiles_list1(string):
    char_list = []
    i = 1
    while i < len(string):
        c = string[i]
        if c.islower():
            char_list.append(string[i-1:i+1])
            i += 1
        else:
            char_list.append(string[i-1])
        i += 1
    if not string[-1].islower():
        char_list.append(string[-1])
    return char_list

def onehot_encode(char_list, smiles_string, length):
    encode_row = lambda char: list(map(int, [c == char for c in smiles_string]))
    ans = np.array(list(map(encode_row, char_list)))
    if ans.shape[1] < length:
        residual = np.zeros((len(char_list), length - ans.shape[1]), dtype=np.int8)
        ans = np.concatenate((ans, residual), axis=1)
    return ans

def smiles_to_onehot(smiles, c_chars, c_length):
    c_ndarray = np.ndarray(shape=(len(smiles), len(c_chars), c_length), dtype=np.float32)
    for i in range(len(smiles)):
        c_ndarray[i, ...] = onehot_encode(c_chars, smiles[i], c_length)
    return c_ndarray

def load_as_ndarray():
    reader = csv.reader(open(folder + "drug_smiles.csv"))
    next(reader, None)
    smiles = np.array(list(reader), dtype=str)
    return smiles

def load_as_ndarray1():
    reader = csv.reader(open(folder + "drug_smiles1.csv"))
    next(reader, None)
    smiles = np.array(list(reader), dtype=str)
    return smiles

def charsets(smiles):
    union = lambda x, y: set(x) | set(y)
    c_chars = list(reduce(union, map(string2smiles_list, list(smiles[:, 2]))))
    i_chars = list(reduce(union, map(string2smiles_list, list(smiles[:, 3]))))
    return c_chars, i_chars

def charsets1(smiles):
    union = lambda x, y: set(x) | set(y)
    print('str',  list(smiles[:, 3]))
    c_chars = list(reduce(union, map(string2smiles_list1, list(smiles[:, 2]))))
    i_chars = list(reduce(union, map(string2smiles_list1, list(smiles[:, 3]))))
    return c_chars, i_chars

def save_drug_smiles_onehot():
    smiles = load_as_ndarray()
    # we will abandon isomerics smiles from now on
    c_chars, _ = charsets(smiles)
    c_length = max(map(len, map(string2smiles_list, list(smiles[:, 2]))))
    
    count = smiles.shape[0]
    drug_names = smiles[:, 0].astype(str)
    drug_cids = smiles[:, 1].astype(int)
    smiles = [string2smiles_list(smiles[i, 2]) for i in range(count)]
    
    canonical = smiles_to_onehot(smiles, c_chars, c_length)
    
    save_dict = {}
    save_dict["drug_names"] = drug_names
    save_dict["drug_cids"] = drug_cids
    save_dict["canonical"] = canonical
    save_dict["c_chars"] = c_chars

    print("drug onehot smiles data:")
    print (drug_names.shape)
    print (drug_cids.shape)
    print (canonical.shape)

    np.save(folder + "drug_onehot_smiles.npy", save_dict)
    print ("finish saving drug onehot smiles data:")
    return drug_names, drug_cids, canonical


def save_drug_smiles_onehot1():
    smiles = load_as_ndarray()
    smiles1 = load_as_ndarray1()
    # we will abandon isomerics smiles from now on
    c_chars, _ = charsets(smiles)
    c_length = max(map(len, map(string2smiles_list, list(smiles[:, 2]))))

    count = smiles1.shape[0]
    drug_names = smiles1[:, 0].astype(str)
    drug_cids = smiles1[:, 1].astype(int)
    smiles1 = [string2smiles_list(smiles1[i, 2]) for i in range(count)]

    canonical = smiles_to_onehot(smiles1, c_chars, c_length)

    save_dict = {}
    save_dict["drug_names"] = drug_names
    save_dict["drug_cids"] = drug_cids
    save_dict["canonical"] = canonical
    save_dict["c_chars"] = c_chars

    print("drug onehot smiles data:")
    print(drug_names.shape)
    print(drug_cids.shape)
    print(canonical.shape)

    np.save(folder + "drug_onehot_smiles1.npy", save_dict)
    print("finish saving drug onehot smiles data:")
    return drug_names, drug_cids, canonical

"""
The following part will prepare the mutation features for the cell.
"""

def save_cell_mut_matrix():
    f = open(folder + "PANCANCER_Genetic_features_cna_Thu Feb  9 12_20_42 2023.csv")
    reader = csv.reader(f)
    next(reader)
    cell_dict = {}
    mut_dict = {}

    matrix_list = []
    organ1_dict = {}
    organ2_dict = {}
    for item in reader:
        cell = item[0]
        mut = item[5]
        organ1_dict[cell] = item[2]
        organ2_dict[cell] = item[3]
        is_mutated = int(item[6])
        if cell in cell_dict:
            row = cell_dict[cell]
        else:
            row = len(cell_dict)
            cell_dict[cell] = row
        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col
        matrix_list.append((row, col, is_mutated))
    
    matrix = np.ones(shape=(len(cell_dict), len(mut_dict)), dtype=np.float32)
    matrix = matrix * -1
    for item in matrix_list:
        matrix[item[0], item[1]] = item[2]

    feature_num = [len(list(filter(lambda x: x >=0, list(matrix[i, :])))) for i in range(len(cell_dict))]
    indics = [i for i in range(len(feature_num)) if feature_num[i]==425]
    matrix = matrix[indics, :]

    inv_cell_dict = {v:k for k,v in cell_dict.items()}
    all_names = [inv_cell_dict[i] for i in range(len(inv_cell_dict))]
    cell_names = np.array([all_names[i] for i in indics])

    inv_mut_dict = {v:k for k,v in mut_dict.items()}
    mut_names = np.array([inv_mut_dict[i] for i in range(len(inv_mut_dict))])
    
    desc1 = []
    desc2 = []
    for i in range(cell_names.shape[0]):
        desc1.append(organ1_dict[cell_names[i]])
        desc2.append(organ2_dict[cell_names[i]])
    desc1 = np.array(desc1)
    desc2 = np.array(desc2)

    save_dict = {}
    save_dict["cell_mut"] = matrix
    save_dict["cell_names"] = cell_names
    save_dict["mut_names"] = mut_names
    save_dict["desc1"] = desc1
    save_dict["desc2"] = desc2

    print ("cell mut data:")
    print (len(all_names))
    print (cell_names.shape)
    print (mut_names.shape)
    print (matrix.shape)
    np.save(folder + "cell_mut_matrix.npy", save_dict)
    print ("finish saving cell mut data:")

    return matrix, cell_names, mut_names


def save_cell_mut_matrix1():
    f = open(folder + "PANCANCER_Genetic_features_cna_Thu Feb  9 12_20_42 2023.csv")
    reader = csv.reader(f)
    next(reader)
    cell_dict = {}
    cell_dict1 = {}
    mut_dict = {}
    mut_dict1 = {}

    matrix_list = []
    organ1_dict = {}
    organ1_dict1 = {}
    organ2_dict = {}
    organ2_dict1 = {}
    for item in reader:
        cell = item[0]
        mut = item[5]
        organ1_dict[cell] = item[2]
        organ2_dict[cell] = item[3]
        is_mutated = int(item[6])
        if cell in cell_dict:
            row = cell_dict[cell]
        else:
            row = len(cell_dict)
            cell_dict[cell] = row
        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

    f1 = open(folder + "PANCANCER_Genetic_feature_Tue Oct 31 03_00_35 20171.csv")
    reader = csv.reader(f1)
    next(reader)

    for item in reader:
        cell = item[0]
        mut = item[5]
        organ1_dict[cell] = item[2]
        organ2_dict[cell] = item[3]
        is_mutated = int(item[6])
        if cell in cell_dict:
            row = cell_dict[cell]
            cell_dict1[cell] = row
        if mut in mut_dict:
            col = mut_dict[mut]
            mut_dict1[mut] = col
        matrix_list.append((row, col, is_mutated))


    matrix = np.ones(shape=(len(cell_dict), len(mut_dict)), dtype=np.float32)
    matrix = matrix * -1
    for item in matrix_list:
        matrix[item[0], item[1]] = item[2]

    feature_num = [len(list(filter(lambda x: x >= 0, list(matrix[i, :])))) for i in range(len(cell_dict))]
    indics = [i for i in range(len(feature_num)) if feature_num[i] == 425]
    matrix = matrix[indics, :]

    inv_cell_dict = {v: k for k, v in cell_dict.items()}
    all_names = [inv_cell_dict[i] for i in range(len(inv_cell_dict))]
    cell_names = np.array([all_names[i] for i in indics])

    inv_mut_dict = {v: k for k, v in mut_dict.items()}
    mut_names = np.array([inv_mut_dict[i] for i in range(len(inv_mut_dict))])

    desc1 = []
    desc2 = []
    for i in range(cell_names.shape[0]):
        desc1.append(organ1_dict[cell_names[i]])
        desc2.append(organ2_dict[cell_names[i]])
    desc1 = np.array(desc1)
    desc2 = np.array(desc2)

    save_dict = {}
    save_dict["cell_mut"] = matrix
    save_dict["cell_names"] = cell_names
    save_dict["mut_names"] = mut_names
    save_dict["desc1"] = desc1
    save_dict["desc2"] = desc2

    print("cell mut data:")
    print(len(all_names))
    print(cell_names.shape)
    print(mut_names.shape)
    print(matrix.shape)
    np.save(folder + "cell_mut_matrix1.npy", save_dict)
    print("finish saving cell mut data:")

    return matrix, cell_names, mut_names


#save_cell_mut_matrix()
#exit()
"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score

"""

def save_drug_cell_matrix1():
    f = open(folder + "PANCANCER_IC_Thu Feb  9 10_47_54 2023.csv")
    reader = csv.reader(f)
    next(reader)

    drug_dict = {}
    cell_dict = {}
    matrix_list = []

    for item in reader:
        drug = item[0]
        cell = item[2]
        
        if drug in drug_dict:
            row = drug_dict[drug]
        else:
            row = len(drug_dict)
            drug_dict[drug] = row
        if cell in cell_dict:
            col = cell_dict[cell]
        else:
            col = len(cell_dict)
            cell_dict[cell] = col
        
        matrix_list.append((row, col, item[7], item[8], item[9], item[10], item[11]))
        
    existance = np.zeros(shape=(len(drug_dict), len(cell_dict)), dtype=np.int32)
    matrix = np.zeros(shape=(len(drug_dict), len(cell_dict), 6), dtype=np.float32)
    for item in matrix_list:
        existance[item[0], item[1]] = 1
        matrix[item[0], item[1], 0] = 1 / (1 + pow(math.exp(float(item[2])), -0.1))
        matrix[item[0], item[1], 1] = float(item[3])
        matrix[item[0], item[1], 2] = float(item[4])
        matrix[item[0], item[1], 3] = float(item[5])
        matrix[item[0], item[1], 4] = float(item[6])
        matrix[item[0], item[1], 5] = math.exp(float(item[2]))

    
    inv_drug_dict = {v:k for k,v in drug_dict.items()}
    inv_cell_dict = {v:k for k,v in cell_dict.items()}
    
    drug_names, drug_cids, canonical = save_drug_smiles_onehot()
    cell_mut_matrix, cell_names, mut_names = save_cell_mut_matrix()
    
    drug_ids = [drug_dict[i] for i in drug_names]
    cell_ids = [cell_dict[i] for i in cell_names]
    sub_matrix = matrix[drug_ids, :][:, cell_ids]
    existance = existance[drug_ids, :][:, cell_ids]
    
    row, col = np.where(existance > 0)
    positions = np.array(zip(row, col))
   
    print (positions.shape)

    save_dict = {}
    save_dict["drug_names"] = drug_names
    save_dict["cell_names"] = cell_names
    save_dict["positions"] = positions
    save_dict["IC50"] = sub_matrix[:, :, 0]
    save_dict["AUC"] = sub_matrix[:, :, 1]
    save_dict["Max_conc"] = sub_matrix[:, :, 2]
    save_dict["RMSE"] = sub_matrix[:, :, 3]
    save_dict["Z_score"] = sub_matrix[:, :, 4]
    save_dict["raw_ic50"] = sub_matrix[:, :, 5]

    print ("drug cell interaction data:")
    print (drug_names.shape)
    print (cell_names.shape)
    print (sub_matrix.shape)
    print (matrix.shape)
   
    np.save(folder + "drug_cell_interaction.npy", save_dict)
    print ("finish saving drug cell interaction data:")
    return sub_matrix


def save_drug_cell_matrix2():
    f = open(folder + "PANCANCER_IC_Thu Feb  9 10_47_54 2023.csv")
    f1 = open(folder + "PANC-10-05.csv")
    reader = csv.reader(f)
    next(reader)

    drug_dict1 = {}
    drug_dict2 = {}
    cell_dict1 = {}
    cell_dict2 = {}
    matrix_list1 = []
    matrix_list2 = []

    for item in reader:
        drug = item[0]
        cell = item[2]

        if drug in drug_dict1:
            row = drug_dict1[drug]
        else:
            row = len(drug_dict1)
            drug_dict1[drug] = row
        if cell in cell_dict1:
            col = cell_dict1[cell]
        else:
            col = len(cell_dict1)
            cell_dict1[cell] = col

    reader = csv.reader(f1)
    next(reader)

    for item in reader:
        drug = item[0]
        cell = item[2]

        if drug in drug_dict2:
            row = drug_dict2[drug]
        else:
            row = drug_dict1[drug]
            drug_dict2[drug] = row
        if cell in cell_dict2:
            col = cell_dict2[cell]
        else:
            col = cell_dict1[cell]
            cell_dict2[cell] = col

        matrix_list2.append((row, col, item[7], item[8], item[9], item[10], item[11]))

    existance = np.zeros(shape=(len(drug_dict1), len(cell_dict1)), dtype=np.int32)
    matrix2 = np.zeros(shape=(len(drug_dict1), len(cell_dict1), 6), dtype=np.float32)
    for item in matrix_list2:
        existance[item[0], item[1]] = 1
        matrix2[item[0], item[1], 0] = 1 / (1 + pow(math.exp(float(item[2])), -0.1))
        matrix2[item[0], item[1], 1] = float(item[3])
        matrix2[item[0], item[1], 2] = float(item[4])
        matrix2[item[0], item[1], 3] = float(item[5])
        matrix2[item[0], item[1], 4] = float(item[6])
        matrix2[item[0], item[1], 5] = math.exp(float(item[2]))

    inv_drug_dict = {v: k for k, v in drug_dict2.items()}
    inv_cell_dict = {v: k for k, v in cell_dict2.items()}

    drug_names, drug_cids, canonical = save_drug_smiles_onehot1()
    cell_mut_matrix, cell_names, mut_names = save_cell_mut_matrix1()

    drug_ids = [drug_dict2[i] for i in drug_names]
    cell_ids = [cell_dict2[i] for i in cell_names]
    sub_matrix = matrix2[drug_ids, :][:, cell_ids]
    existance = existance[drug_ids, :][:, cell_ids]

    row, col = np.where(existance > 0)
    positions = np.array(zip(row, col))

    print(positions.shape)

    save_dict1 = {}
    save_dict1["drug_names"] = drug_names
    save_dict1["cell_names"] = cell_names
    save_dict1["positions"] = positions
    save_dict1["IC50"] = sub_matrix[:, :, 0]
    save_dict1["AUC"] = sub_matrix[:, :, 1]
    save_dict1["Max_conc"] = sub_matrix[:, :, 2]
    save_dict1["RMSE"] = sub_matrix[:, :, 3]
    save_dict1["Z_score"] = sub_matrix[:, :, 4]
    save_dict1["raw_ic50"] = sub_matrix[:, :, 5]

    print("drug cell interaction data:")
    print(drug_names.shape)
    print(cell_names.shape)
    print(sub_matrix.shape)
    print(matrix.shape)

    np.save(folder + "drug_cell_interaction1.npy", save_dict1)
    print("finish saving drug cell interaction data:")
    return sub_matrix
    
save_drug_cell_matrix1()
save_drug_cell_matrix2()
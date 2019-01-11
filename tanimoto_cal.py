import re
import copy
import pandas as pd
import numpy as np


fingerprints_file_1 = "sim_test/fingerprints_strings1.txt"
fingerprints_file_2 = "sim_test/fingerprints_strings2.txt"
drug_names_file_s   = "sim_test/drugs_names.txt"

col_names       = ["chemical_name"]
drugs_names_df_s = pd.read_csv(drug_names_file_s, names=col_names)

size_drugs = len(drugs_names_df_s)

col_names_s        = ["chemical_name","fingerprint"]
fingerprints_df_1  = pd.read_csv(fingerprints_file_1, names=col_names_s)
fingerprints_df_2  = pd.read_csv(fingerprints_file_2, names=col_names_s)

frames = [fingerprints_df_1, fingerprints_df_2]
fingerprints_df_s = pd.concat(frames,ignore_index=True)

similarity_matrix = np.zeros(shape=(size_drugs,size_drugs))

upper_indexes = np.triu_indices(size_drugs,k=1)
num_indexes = len(upper_indexes[0])

for i in range(num_indexes):
    x = upper_indexes[0][i]
    y = upper_indexes[1][i]
    fingerprint1 = fingerprints_df_s.fingerprint[x]
    fingerprint2 = fingerprints_df_s.fingerprint[y]
    fingerprintsAND = bin(int(str(fingerprint1),2) & int(str(fingerprint2),2))[2:].zfill(len(fingerprint1)).count("1")
    fingerprintsXOR = bin(int(fingerprint1,2) ^ int(fingerprint2,2))[2:].zfill(len(fingerprint1)).count("1")
    tanimoto = float(fingerprintsAND)/(fingerprintsXOR+ fingerprintsAND) 
    similarity_matrix[x][y] = tanimoto

np.savetxt('similarity_test1.txt',similarity_matrix, fmt='%.6f', delimiter=',')

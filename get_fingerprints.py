import re


file_fingerprints   = 'ikey2ecfp4_sample2.tsv'
file_drug_names     = 'drug_names1.txt'

def drugs_to_list(file_handle, drug_list):    
    with open(file_handle) as fh:
        content = fh.readlines()
        content = [x.strip() for x in content]
        for line in content:
            drug_list.append(line)
    fh.close()


def get_fingerprints(file_handle,drugs_fingerprints_dict, drugs_list):
    with open(file_handle) as fh:
        j=0
        content = fh.readlines()
        content = [x.strip() for x in content]
        for line in content:
            result = re.split(r'[,\t]\s*',line)
            drug_name = result[0]
            if drug_name in drugs_list:
                j=j+1
                for i in range(1,1025):
                    drugs_fingerprints_dict[drug_name][i-1] = result[i]
    fh.close()
    print(j)




drugs_list = list()
drugs_to_list(file_drug_names,drugs_list)


drugs_fingerprints_dict = {}
for i in range(len(drugs_list)):
    drugs_fingerprints_dict[drugs_list[i]] = [0] * 1024

get_fingerprints(file_fingerprints,drugs_fingerprints_dict ,drugs_list )






with open('drug_fingerprints.txt', 'w') as f:
    for key, value in drugs_fingerprints_dict.items():
        f.write(key)
        f.write(',')
        for i in range(len(value)):
            f.write(str(value[i]))
            f.write(",")
        f.write('\n')


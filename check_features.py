import numpy as np
import pickle


#hettemp = '/het/p1/ranit/qg/disco_ffs/temp/'
hettemp = './temp/'
list_vars = []

i='bip_m_pt'
with open(hettemp+'variables_7k_iter_'+str(i)+'.txt','rb') as fp:
	variables = pickle.load(fp)
#	print(variables)
#	for j in variables[0:7]:
#		list_vars.append(j)

with open(hettemp+'variables1_7k_iter_'+str(i)+'.txt','rb') as fp:
	variables1 = pickle.load(fp)

print(variables)
#variables['bip']=variables['bip'][0:7]
#print(variables['mf_s2'][0:17])

#variables1['efp']=variables1['efp'][0:10]
#print(variables['bip'][0:7])
#print(variables1['bip'][0:7])


#i='qg_7k_efps_test_2'
#print(variables1['efp'][0:10])
#print(variables1)
#myset = set(list_vars)
#print(len(list(myset)))
#with open(hettemp+'variables1_7k.txt','rb') as fp:
	#variables1 = pickle.load(fp)

#variables = [3407, 7095, 1476, 1473, 3921, 1471]
#variables1 = [3407, 7095, 1476, 3923, 1473, 3921, 1471]
#variables1 = [3407, 7095, 1476, 1473, 3921, 1471]
#print(variables1[0:11])
#print(variables[0:10])
#print(len(variables[0:10]))

#print(len(variables))
#print(variables[0:2])
#print(len(variables[0:11]))

#variables = variables[0:15]
#variables1 = variables1[0:16]
#print(variables)
#print(variables1)
#print(variables[0:13])
#print(variables1[0:14])
#print(len(variables))

#with open(hettemp+'variables_7k_iter_'+str(i)+'.txt','wb') as fp:
#	pickle.dump(variables,fp)
#with open(hettemp+'variables1_7k.txt','wb') as fp:
#	pickle.dump(variables1,fp)

#with open(hettemp+'variables1_7k_iter_'+str(i)+'.txt','wb') as fp:
#	pickle.dump(variables1,fp)



import numpy as np
import random
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect


# define functions to map RDKit mol objects to list of atomic invariants, i.e. to a list of their initial atom feature vectors
# using ecfp_invariants will later lead to the standard ECFP, but one can also experiment with other atom invariants

def random_invariants(mol):
	invs = random.sample(range(1, 1000000), mol.GetNumAtoms())
	return invs

def uniform_invariants(mol):
	invs = [1]*mol.GetNumAtoms()
	return invs

def atomic_number_invariants(mol):
	invs = []
	for atom in mol.GetAtoms():
		invs.append(atom.GetAtomicNum())
	return invs

def ecfp_invariants(mol):
    invs = []
    for k in range(mol.GetNumAtoms()):
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius = 0, atomInvariantsGenerator = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership = True))
        fingerprint = fp_gen.GetSparseCountFingerprint(mol, fromAtoms = [k])        
        invs.append(list(fingerprint.GetNonzeroElements().keys())[0])
    return invs

def fcfp_invariants(mol):
    invs = []
    for k in range(mol.GetNumAtoms()):
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius = 0, atomInvariantsGenerator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
        fingerprint = fp_gen.GetSparseCountFingerprint(mol, fromAtoms = [k])        
        invs.append(list(fingerprint.GetNonzeroElements().keys())[0])
    return invs

def combined_invariants(invariant_func_1, invariant_func_2):
	def combined_invariants(mol):
		invs_1 = invariant_func_1(mol)
		invs_2 = invariant_func_2(mol)
		invs = [hash((i_1, i_2)) % 10000001 for (i_1, i_2) in list(zip(invs_1, invs_2))]
		return invs
	return combined_invariants

def one_hot_vec(dim, k):
	vec = np.zeros(dim)
	vec[k] = 1
	return vec.astype(int)

# define a function that mapes a smiles string to a set of integer identifiers of circular ECFP substructures
# the output of this function can be seen as a set representation of the input compound, whereby each element of the output set corresponds to a circular substructure that can be found in the input compound
def ecfp_atom_ids_from_smiles(smiles, ecfp_settings):
	mol = Chem.MolFromSmiles(smiles)
	info_dict = {}
	fp = GetMorganFingerprint(mol,
							radius = ecfp_settings["radius"],
							useCounts = ecfp_settings["use_counts"],
							invariants = ecfp_settings["mol_to_invs_function"](mol), 
							useChirality = ecfp_settings["use_chirality"], 
							useBondTypes = ecfp_settings["use_bond_invs"], 
							bitInfo = info_dict)
	return (UIntSparseIntVect.GetNonzeroElements(fp), info_dict)

# smiles = "CC(=O)NCCC1=CNc2c1cc(OC)cc2"

# ecfp_settings = {"mol_to_invs_function": ecfp_invariants,
# 				 "radius": 2,
# 				 "pool_method": "sorted", # choose either "hashed" to later get the standard hashed ECFP-vector, or choose "sorted" to get the Sort & Slice ECFP-vector
# 				 "dimension": 1024,
# 				 "use_bond_invs": True,
# 				 "use_chirality": False,
# 				 "use_counts": False}

# print(ecfp_atom_ids_from_smiles(smiles, ecfp_settings)[0], "\n \n") # this outputs a dictionary whose keys are the detected substructure identifiers and whose values are their respective counts in the molecule, but since we used use_counts = False, the count is always 1
# print(ecfp_atom_ids_from_smiles(smiles, ecfp_settings)[1]) # this is a dictionary that gives information about where each substructure is located in the input compound


# this function takes a list of smiles strings x_smiles = [smiles_1, smiles_2, ....] and a dictionary of ecfp_settings as input
# it outputs a function that maps substructure identifiers (i.e. atom ids) to one hot encodings, whereby the components of the one-hot encoding are sorted according to the frequency of the substructures in the training set
# i.e. the first component corresponds to the substructure that occurs in the most training compounds and so on
# if the required dimension specified in ecfp_settings is larger than the number of training substrutures, then we pad with 0s to reach the desired length
# the very last component of the one-hot encoding corresponds to an "unknown" substructure, i.e. to any substructure (i.e. atom id) that does not occur in the traning set

def create_ecfp_atom_id_one_hot_encoder_frequency(x_smiles, ecfp_settings):
	# preallocate data structures
	atom_id_set = set()
	atom_id_to_support_list_with_counts = {}

	# create set of all occuring atom ids and associated feature matrix with support columns
	for (k, smiles) in enumerate(x_smiles):
		(current_atom_id_to_count, current_info) = ecfp_atom_ids_from_smiles(smiles, ecfp_settings)
		atom_id_set = atom_id_set.union(set(current_atom_id_to_count))

		for atom_id in set(current_atom_id_to_count):
			atom_id_to_support_list_with_counts[atom_id] = atom_id_to_support_list_with_counts.get(atom_id, [0]*len(x_smiles))
			atom_id_to_support_list_with_counts[atom_id][k] = current_atom_id_to_count[atom_id]

	# binarise support list so that it only indicates presence/absence of fragments in training compounds (in case we used counts before)
	atom_id_to_support_list = {atom_id: [1 if b > 0 else 0 for b in support_list_with_counts] for (atom_id, support_list_with_counts) in atom_id_to_support_list_with_counts.items()}

	print("Number of unique substructures = ", len(atom_id_set))

	atom_id_to_support_cardinality = {atom_id: sum(support_list) for (atom_id, support_list) in atom_id_to_support_list.items()}
	atom_id_list_sorted = sorted(list(atom_id_set), key = lambda atom_id: atom_id_to_support_cardinality[atom_id], reverse = True)

	final_atom_id_list = atom_id_list_sorted[0: ecfp_settings["dimension"]]

	zero_padding_dim = int(-min(len(final_atom_id_list) - ecfp_settings["dimension"], 0)) + 1
	final_atom_id_list_to_one_hot_vecs = dict([(atom_id, one_hot_vec(len(final_atom_id_list) + zero_padding_dim, k)) for (k, atom_id) in enumerate(final_atom_id_list)])

	def atom_id_one_hot_encoder(atom_id):
	    other_vec = one_hot_vec(len(final_atom_id_list) + zero_padding_dim, len(final_atom_id_list) + zero_padding_dim - 1)
	    return final_atom_id_list_to_one_hot_vecs.get(atom_id, other_vec)
	return atom_id_one_hot_encoder



# load some data and create an atom id one-hot encoder
# dataframe = pd.read_csv("moleculenet_lipophilicity/" + "clean_data.csv", sep = ",")
# display(dataframe)
# x_smiles = np.reshape(dataframe["smiles"].values, (len(dataframe), 1))[:,0]

# atom_id_one_hot_encoder = create_ecfp_atom_id_one_hot_encoder_frequency(x_smiles, ecfp_settings)


# a function that takes a smiles string, and ecfp_settings dictionary and the afforementioned atom_id_one_hot_encoder
# it outputs a representation of the input compound as a set of embedded substructures, whereby atom_id_one_hot_encoder is used as a substructure embedding
# technically, we use multisets instead of sets so that everything also works if we use ecfps with counts
# however, in the standard case where we use ecfps without counts, the representations can be thought of simply as normal sets
# the multiset is represented as a numpy array whose rows corresponds to substructure embeddings
def create_ecfp_vector_multiset(smiles, ecfp_settings, atom_id_one_hot_encoder):
    atom_id_dict = ecfp_atom_ids_from_smiles(smiles, ecfp_settings)[0]
    atom_id_list = []
    for key in atom_id_dict:
        atom_id_list += [key]*atom_id_dict[key]
    vector_multiset = np.array([atom_id_one_hot_encoder(atom_id) for atom_id in atom_id_list])
    vector_multiset = np.delete(vector_multiset, np.where(vector_multiset[:,-1] == 1)[0], axis = 0)[:,0:-1] # we delete the final column and the rows in the array that correspond to unknowns substructures that do not appear in the training set
    return vector_multiset

# example HERES WHERE THE SS FINGERPRINT IS
# vector_multiset = create_ecfp_vector_multiset(x_smiles[10], ecfp_settings, atom_id_one_hot_encoder)

# print(vector_multiset.shape)

# # note that if we sum up the rows of vector_multiset we already get the binary sort and slice fingerprint for the corresponding input compound

# print(np.sum(vector_multiset, axis = 0).shape)
# print("Sort and Slice ECFP-fingerprint = ", np.sum(vector_multiset, axis = 0))


# finally, we plug everything together to define a function that creates an ecfp featuriser, based on either the standard hashing procedure or on sort and slice
# this function takes a dictionary with ecfp setting and a list of smiles strings that in a machine-learning context would correspond to the training set
# it outputs a function called "featuriser"
# if ecfp_settings["pool_method"] == "hashed", then the featuriser transforms a list of smiles strings into a numpy array whose rows correspond to the standard hashed ECFPs
# if ecfp_settings["pool_method"] == "sorted", then the featuriser transforms a list of smiles strings into a numpy array whose rows correspond to the Sort and Slice ECFPs


def create_ecfp_featuriser(ecfp_settings=None, x_smiles_train=None):
    # if ecfp_settings["pool_method"] == "hashed":
    #     def featuriser(x_smiles):
    #         x_mol = [Chem.MolFromSmiles(smiles) for smiles in x_smiles]
    #         X_fp = np.array([Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
	# 																			 radius = ecfp_settings["radius"],
	# 																			 nBits = ecfp_settings["dimension"],
	# 																			 invariants = ecfp_settings["mol_to_invs_function"](mol),
	# 																			 useBondTypes = ecfp_settings["use_bond_invs"],
	# 																			 useChirality = ecfp_settings["use_chirality"]) for mol in x_mol])
    #         return X_fp
    #     return featuriser
    # elif ecfp_settings["pool_method"] == "sorted":
    atom_id_one_hot_encoder = create_ecfp_atom_id_one_hot_encoder_frequency(x_smiles_train, ecfp_settings)
    def featuriser(x_smiles):
        X_fp = np.zeros((len(x_smiles), ecfp_settings['dimension']))
        for (k, smiles) in enumerate(x_smiles):
            mol = Chem.MolFromSmiles(smiles)
            X_fp[k,:] = np.sum(create_ecfp_vector_multiset(smiles, ecfp_settings, atom_id_one_hot_encoder), axis = 0)
        return X_fp
    return featuriser

# create hashed fingerprints

# featuriser = create_ecfp_featuriser(ecfp_settings, x_smiles_train = None)

# X_fps = featuriser(x_smiles)

# print(X_fps.shape)

# # create sort and slice fingerprints, whereby we use the first k smiles in our list x_smiles as the training set

# k = 50000

# ecfp_settings = {"mol_to_invs_function": ecfp_invariants,
#                  "radius": 2,
#                  "pool_method": "sorted", # choose either "hashed" to later get the standard hashed ECFP-vector, or choose "sorted" to get the Sort & Slice ECFP-vector
#                  "dimension": 1024,
#                  "use_bond_invs": True,
#                  "use_chirality": False,
#                  "use_counts": False}

# featuriser = create_ecfp_featuriser(ecfp_settings, x_smiles_train = x_smiles[0:k])

# X_fps = featuriser(x_smiles)

# print(X_fps.shape)

import numpy as np
import torch
from torch.autograd import Variable


dim = 50
arr_of_roles = ["P1","P2","P3"] # missing: create array of vectors of roles. The array should have 3 elements. Every element is a vector of dimension "dim"
arr_of_atoms_for_args = ["a","b","d","e","f","g"]
arr_of_atoms_for_ops = ["C","R"]




def get_dict_of_rand_vecs_from_array(arr):
    dict = {}
    for i,a in enumerate(arr):
        dict[a] = torch.rand(dim)

    return dict

dict_of_args_vectors = get_dict_of_rand_vecs_from_array(arr_of_atoms_for_args)
dict_of_ops_vectors = get_dict_of_rand_vecs_from_array(arr_of_atoms_for_ops)
dict_of_role_vectors = get_dict_of_rand_vecs_from_array(arr_of_roles)
dict_of_pointers = {}

dict_of_atom_vectors = {**dict_of_args_vectors,**dict_of_ops_vectors}


def get_mats_from_arr(arr, command_number):
    arr_of_mats = []
    arr_of_diags = []
    arr_of_symbols_for_command = []
    arr_of_commands = []
    final_mat = torch.zeros(dim,dim) # missing: matrix of zeros of dimension dim x dim
    for i,el in enumerate(arr):
        vec_for_role = dict_of_role_vectors[arr_of_roles[i]]

        if isinstance(el, list): # missing: check if type of element is array
            command_number += 1
            diags, mats, commands = get_mats_from_arr(el, command_number)
            arr_of_mats.append(mats)
            arr_of_diags.append(diags)
            pointer_name = "T"+str(command_number)
            arr_of_symbols_for_command.append(pointer_name)
            arr_of_commands.append(commands)
            dict_of_pointers[pointer_name] = diags[0]
            vec_for_el = diags[0]
        else: # it's atom
            vec_for_el = dict_of_atom_vectors[el]
            arr_of_symbols_for_command.append(el)
        final_mat += torch.ger(vec_for_el, vec_for_role) # missing: outer product of role and element
    diag_of_final_mat = torch.diag(final_mat) # missing: get diagonal of matrix
    arr_of_mats.insert(0, final_mat) # prepend to the array
    arr_of_diags.insert(0, diag_of_final_mat)
    arr_of_commands.insert(0, arr_of_symbols_for_command)
    return arr_of_diags, arr_of_mats, arr_of_commands


example = ["C","a",["R",["C","b",["R","d","e"]],["C","f","g"]]]

arr_of_diags, arr_of_mats, commands = get_mats_from_arr(example,1)
dict_of_pointers["T1"] = arr_of_diags[0]





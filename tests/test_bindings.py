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
        vec = torch.rand(dim)
        norm = torch.norm(vec,2)
        dict[a] = torch.div(vec,norm)
        print("norm",torch.norm(dict[a]))

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
    nodes_traversed = 0
    for i,el in enumerate(arr):
        vec_for_role = dict_of_role_vectors[arr_of_roles[i]]

        if isinstance(el, list): # missing: check if type of element is array
            command_number+=1
            pointer_name = "T"+str(command_number)
            diags, mats, commands, nodes_traversed_in_this_tree = get_mats_from_arr(el, command_number)
            arr_of_mats.append(mats)
            arr_of_diags.append(diags)
            nodes_traversed += nodes_traversed_in_this_tree
            command_number += nodes_traversed-1
            arr_of_symbols_for_command.append(pointer_name)
            arr_of_commands.append(commands)
            dict_of_pointers[pointer_name] = diags[0]
            vec_for_el = diags[0]
        else: # it's atom
            vec_for_el = dict_of_atom_vectors[el]
            arr_of_symbols_for_command.append(el)
        final_mat += torch.addr(torch.zeros(dim,dim),vec_for_el, vec_for_role) # missing: outer product of role and element
    diag_of_final_mat = torch.diag(final_mat) # missing: get diagonal of matrix
    normed_diag_of_final_mat = torch.div(diag_of_final_mat,torch.norm(diag_of_final_mat,2)) # normalize it
    arr_of_mats.insert(0, final_mat) # prepend to the array
    arr_of_diags.insert(0, normed_diag_of_final_mat)
    arr_of_commands.insert(0, arr_of_symbols_for_command)
    return arr_of_diags, arr_of_mats, arr_of_commands, nodes_traversed+1

example = ["C","a",["R",["C","b",["R","d","e"]],["C","f","g"]]]

arr_of_diags, arr_of_mats, commands,nodes_traversed = get_mats_from_arr(example,1)
dict_of_pointers["T1"] = arr_of_diags[0]

def simplify_array(arr):
    simple_arr=[]
    for a in arr:
        if isinstance(a,list):
            simple_arr += simplify_array(a)
        else:
            simple_arr.append(a)
    return simple_arr

def simplify_array_of_commands(arr):
    simple_arr = []
    for a in arr:
        if all(isinstance(s, str) for s in a):
            simple_arr.append(a)
        else:
            simple_arr += simplify_array_of_commands(a)
    return simple_arr


pure_arr_of_diags = simplify_array(arr_of_diags)
pure_arr_of_mats = simplify_array(arr_of_mats)
pure_arr_of_commands = simplify_array_of_commands(commands)

for d in pure_arr_of_diags:
    print('max_diag',torch.max(d))
# compute the Moore-Penrose inverse
def get_unbinders_from_mat(mat):
    mat_np = mat.numpy()
    inverse_np = np.linalg.pinv(mat_np)
    return torch.from_numpy(inverse_np)


mat_from_diags = torch.stack(pure_arr_of_diags)
inverse_of_diags = get_unbinders_from_mat(mat_from_diags)

arr_of_role_vecs = []
for r in arr_of_roles:
    arr_of_role_vecs.append(dict_of_role_vectors[r])

mat_from_roles = torch.stack(arr_of_role_vecs)
inverse_of_roles = get_unbinders_from_mat(mat_from_roles)


tensor_subcommands = torch.stack(pure_arr_of_mats,2)

memory_tensor = torch.matmul(tensor_subcommands,mat_from_diags)

restored_subcomands = torch.matmul(memory_tensor, inverse_of_diags)

all_used_symbols = {**dict_of_atom_vectors,**dict_of_pointers}


C = tensor_subcommands - restored_subcomands
print(C)

# test wether the matrices are good
# for i,c in enumerate(pure_arr_of_commands):
#     mat = pure_arr_of_mats[i]
#
#     mat_for_command = torch.zeros(dim,dim)
#     for i,t in enumerate(c):
#         vec_for_command = all_used_symbols[t]
#         vec_for_role = dict_of_role_vectors[arr_of_roles[i]]
#         mat_for_command += torch.addr(torch.zeros(dim,dim),vec_for_command,vec_for_role)
#
#     difference = mat - mat_for_command
#     print(c)
#     print("diffenrece",difference.sum())




for i,c in enumerate(pure_arr_of_commands):
    mat = pure_arr_of_mats[i].numpy()

    mat_for_command = restored_subcomands[:,:,i].numpy()
    print(c)
    output = np.subtract(mat_for_command,mat)
    print(output.sum())
#un_role = torch.matmul(restored_subcomands,inverse_of_roles)
import numpy as np
import torch
from torch.autograd import Variable


class SymbolManipulator():
    def __init__(self, dim, arr_of_roles, arr_of_atoms_for_args, arr_of_atoms_for_ops):
        self.dim = dim
        self.arr_of_roles = arr_of_roles
        self.arr_of_atoms_for_args = arr_of_atoms_for_args
        self.arr_of_atoms_for_ops = arr_of_atoms_for_ops

        self.dict_of_args_vectors = self.get_dict_of_rand_vecs_from_array(self.arr_of_atoms_for_args)
        self.dict_of_ops_vectors = self.get_dict_of_rand_vecs_from_array(self.arr_of_atoms_for_ops)
        self.dict_of_role_vectors = self.get_dict_of_rand_vecs_from_array(self.arr_of_roles)
        self.dict_of_pointers = {}
        self.dict_of_atom_vectors = {**self.dict_of_args_vectors,**self.dict_of_ops_vectors}

        # create array of role vectors which will be used for creation of unbinding vectors
        self.pure_arr_of_role_vecs = []
        for r in self.arr_of_roles:
            self.pure_arr_of_role_vecs.append(self.dict_of_role_vectors[r])


    def get_dict_of_rand_vecs_from_array(self, arr):
        dict = {}
        for i,a in enumerate(arr):
            vec = torch.rand(dim)
            norm = torch.norm(vec,2)
            dict[a] = torch.div(vec,norm)
        return dict

    def create_memory_for_commands(self, arr_of_commands):
        arr_of_diags, arr_of_mats, commands, self.nodes_traversed = self.get_mats_from_arr(example, 1)
        self.dict_of_pointers["T1"] = arr_of_diags[0]

        self.all_used_symbols = {**self.dict_of_atom_vectors,**self.dict_of_pointers}

        self.pure_arr_of_diags = self.simplify_array(arr_of_diags)
        self.pure_arr_of_mats = self.simplify_array(arr_of_mats)
        self.pure_arr_of_commands = self.simplify_array_of_commands(commands)


        #create array of arrays with vectors for the symbols
        self.pure_arr_of_vecs_for_commands = []
        for com in self.pure_arr_of_commands:
            arr_for_command = []
            for sym in com:
                arr_for_command.append(S_m.all_used_symbols[sym])
            self.pure_arr_of_vecs_for_commands.append(arr_for_command)

        #create memory tensor
        self.mat_from_diags = torch.stack(self.pure_arr_of_diags)
        self.inverse_mat_of_diags = self.get_unbinders_from_mat(self.mat_from_diags)
        self.mat_from_roles = torch.stack(self.pure_arr_of_role_vecs)
        self.inverse_mat_of_roles = self.get_unbinders_from_mat(self.mat_from_roles)
        tensor_subcommands = torch.stack(self.pure_arr_of_mats,2)
        self.memory_tensor = torch.matmul(tensor_subcommands, self.mat_from_diags)

        #create circular matrices
        self.circular_diags = self.create_circular(self.mat_from_diags, self.inverse_mat_of_diags)
        self.circular_roles = self.create_circular(self.mat_from_roles, self.inverse_mat_of_roles)


    def create_circular(self,orig,inverse):
        np_inverse = inverse.numpy()
        np_rolled_inv = np.roll(np_inverse,np_inverse.shape[1]-1,1)
        rolled_inv = torch.from_numpy(np_rolled_inv)
        circular = torch.matmul(rolled_inv, orig)
        return circular


    def get_mats_from_arr(self, arr, command_number):
        arr_of_mats = []
        arr_of_diags = []
        arr_of_symbols_for_command = []
        arr_of_commands = []
        final_mat = torch.zeros(self.dim,self.dim) # missing: matrix of zeros of dimension dim x dim
        nodes_traversed = 0
        for i,el in enumerate(arr):
            vec_for_role = self.dict_of_role_vectors[arr_of_roles[i]]

            if isinstance(el, list): # missing: check if type of element is array
                command_number+=1
                pointer_name = "T"+str(command_number)
                diags, mats, commands, nodes_traversed_in_this_tree = self.get_mats_from_arr(el, command_number)
                arr_of_mats.append(mats)
                arr_of_diags.append(diags)
                nodes_traversed += nodes_traversed_in_this_tree
                command_number += nodes_traversed-1
                arr_of_symbols_for_command.append(pointer_name)
                arr_of_commands.append(commands)
                self.dict_of_pointers[pointer_name] = diags[0]
                vec_for_el = diags[0]
            else: # it's atom
                vec_for_el = self.dict_of_atom_vectors[el]
                arr_of_symbols_for_command.append(el)
            final_mat += torch.addr(torch.zeros(dim,dim),vec_for_el, vec_for_role) # missing: outer product of role and element
        diag_of_final_mat = torch.diag(final_mat) # missing: get diagonal of matrix
        normed_diag_of_final_mat = torch.div(diag_of_final_mat,torch.norm(diag_of_final_mat,2)) # normalize it
        arr_of_mats.insert(0, final_mat) # prepend to the array
        arr_of_diags.insert(0, normed_diag_of_final_mat)
        arr_of_commands.insert(0, arr_of_symbols_for_command)
        return arr_of_diags, arr_of_mats, arr_of_commands, nodes_traversed+1

    def simplify_array(self,arr):
        simple_arr=[]
        for a in arr:
            if isinstance(a,list):
                simple_arr += self.simplify_array(a)
            else:
                simple_arr.append(a)
        return simple_arr

    def simplify_array_of_commands(self,arr):
        simple_arr = []
        for a in arr:
            if all(isinstance(s, str) for s in a):
                simple_arr.append(a)
            else:
                simple_arr += self.simplify_array_of_commands(a)
        return simple_arr

    # compute the Moore-Penrose inverse
    def get_unbinders_from_mat(self,matice):
        mat_np = matice.numpy()
        inverse_np = np.linalg.pinv(mat_np)
        return torch.from_numpy(inverse_np)

    def unbind_symbols_in_chunk(self,chunk):
        symbols = torch.matmul(chunk,self.inverse_mat_of_roles)
        return symbols

    def unbind_chunk(self,att_vec):
        un_att = torch.mul(self.memory_tensor, att_vec)
        un_att = torch.sum(un_att,2)
        return un_att

dim = 50
arr_of_roles = ["P1","P2","P3"] # missing: create array of vectors of roles. The array should have 3 elements. Every element is a vector of dimension "dim"
arr_of_atoms_for_args = ["a","b","d","e","f","g"]
arr_of_atoms_for_ops = ["C","R"]

example = ["C","a",["R",["C","b",["R","d","e"]],["C","f","g"]]]

S_m = SymbolManipulator(dim, arr_of_roles, arr_of_atoms_for_args, arr_of_atoms_for_ops)
S_m.create_memory_for_commands(example)


att_vec = S_m.inverse_mat_of_diags[:, 0]
for a in range(5):
    chunk = S_m.unbind_chunk(att_vec)
    orig_chunk = S_m.pure_arr_of_mats[a]
    B = orig_chunk-chunk
    tensor_to_remove = torch.mul(torch.unsqueeze(chunk,2),torch.unsqueeze(torch.unsqueeze(S_m.pure_arr_of_diags[a],0),1))
    dot = torch.dot(S_m.pure_arr_of_diags[a],att_vec)
    un_att = torch.mul(tensor_to_remove, att_vec)
    un_att = torch.sum(un_att, 2)
    C=un_att-chunk
    S_m.memory_tensor -= tensor_to_remove
    symbols = S_m.unbind_symbols_in_chunk(chunk)
    orig = torch.stack(S_m.pure_arr_of_vecs_for_commands[a],1)
    diff = orig-symbols
    print(S_m.pure_arr_of_commands[a])
    print((diff).sum())
    att_vec = torch.mv(S_m.circular_diags,att_vec)








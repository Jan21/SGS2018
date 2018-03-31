arr_of_mats = []
arr_of_diags = []

dim = 100
arr_of_role_vecs = [] # missing: create array of vectors of roles. The array should have 3 elements. Every element is a vector of dimension "dim"
dict_of_el_vectors={
    "c":create_rand_vector(), # missing: assignment of vectors to atoms/elements
    "a":create_rand_vector()
}


def get_mats_from_arr(arr):
    arr_of_mats = []
    arr_of_diags = []
    final_mat = empty_mat() # missing: matrix of zeros of dimension dim x dim
    for i,el in enumerate(arr):
        vec_for_role = arr_of_role_vecs[i]

        if el is type(array): # missing: check if type of element is array
            diags,mats = get_mats_from_arr(el)
            arr_of_mats.append(mats)
            arr_of_diags.append(diags)
            vec_for_el = diags[0]
        else: # it's atom
            vec_for_el = dict_of_el_vectors[el]
        final_mat += create_outer_product(vec_for_role, vec_for_el) # missing: outer product of role and element
    diag_of_final_mat = diagonal(final_mat) # missing: get diagonal of matrix
    arr_of_mats.insert(0, final_mat) # prepend to the array
    arr_of_diags.insert(0, diag_of_final_mat)
    return arr_of_diags, arr_of_mats


example = ["C","a",["R",["C","b",["R","d","e"]],["C","f","g"]]]

arr_of_diags, arr_of_mats = get_mats_from_arr(example)
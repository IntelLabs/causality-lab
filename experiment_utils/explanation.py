from itertools import combinations


def exhaustive_search_explanation(relevant_tokens_pos, target_token, is_explanation, search_minimal=True):
    """
    Exhaustive search for the minimal subset of tokens that complies with the explanation-definition
    for the target node.

    :param relevant_tokens_pos: indexes of tokens from which to search for a minimal subset of explanation.
    :param target_token: the index of the target token needed to be explained.
    :param is_explanation: a function that tests if a subset is an explanation for a target token. It contains the
        definition of explanation for the target token. It takes as input:
        tokens_list: a list of tokens
        explanation_token_pos: indexes of those tokens to be considered as explanation
        target_pos: index of the target node to be explained
    :param search_minimal:
    :return:
    """
    found_flag = False
    minimal_explanations = []
    n_nodes = len(relevant_tokens_pos)
    for set_size in range(1, n_nodes):
        for explanation_subset in combinations(relevant_tokens_pos, set_size):
            if target_token in explanation_subset:
                continue
            if is_explanation(list(explanation_subset), target_token):
                minimal_explanations.append(explanation_subset)
                found_flag = True
        if search_minimal and found_flag:
            break  # do not search for larger explaining sets
    return minimal_explanations
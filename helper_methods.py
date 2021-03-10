def levenshtein_distance(str1, str2):
    matrix = [0]
    for i in range(len(str1)):
        matrix.append(i + 1)
    matrix = [matrix]
    for j in range(len(str2)):
        matrix.append([j + 1] * (len(str1) + 1))

    for i in range(len(str1)):
        for j in range(len(str2)):
            if str2[j] == str1[i]:
                matrix[j + 1][i + 1] = matrix[j][i]
            else:
                matrix[j + 1][i + 1] = min(matrix[j][i + 1], matrix[j + 1][i], matrix[j][i]) + 1

    return matrix[-1][-1]


def check_spelling_helper(word: str, set_of_words):
    the_most_right_name_edit = float("inf")
    proposed_name = None
    for reference in set_of_words:
        min_edit = levenshtein_distance(word, reference)
        if min_edit < the_most_right_name_edit:
            the_most_right_name_edit = min_edit
            proposed_name = reference
    return proposed_name

print(round(pow(64, 2/3), 4))
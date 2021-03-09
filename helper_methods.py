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

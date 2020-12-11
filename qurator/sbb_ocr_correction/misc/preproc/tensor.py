import torch


def letterToIndex(letter, char_set):
    return char_set.find(letter)


def lineToTensor(line, char_set):
    tensor = torch.zeros(len(line), 1, len(char_set))
    for i, char in enumerate(line):
        if char in char_set:
            tensor[i][0][letterToIndex(char)] = 1
    return tensor


def pairToTensor(pair, char_set):
    input_tensor = lineToTensor(pair[0], char_set)
    output_tensor = lineToTensor(pair[1], char_set)
    return (input_tensor, output_tensor)

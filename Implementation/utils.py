import torch


def metrics(output, y, num_classes, use_gpu=False, device=None):
    predict = torch.argmax(output, dim=1)
    true_positive = torch.zeros(num_classes)
    false_positive = torch.zeros(num_classes)
    false_negative = torch.zeros(num_classes)
    if use_gpu:
        true_positive = true_positive.to(device)
        false_positive = false_positive.to(device)
        false_negative = false_negative.to(device)
    filter_255 = y != 255  # TODO
    """In the context of the Pascal VOC challenge, the label 255 does not represent the background. Instead, it is typically used to denote regions that should be ignored during training and evaluation. These could be ambiguous parts of the image, borders, or any other segments that do not belong to any object class of interest."""

    for i in range(num_classes):
        positive_i = predict == i
        true_i = y == i
        true_positive[i] += torch.sum(positive_i & true_i)
        false_positive[i] += torch.sum(positive_i & ~true_i & filter_255)
        false_negative[i] += torch.sum(~positive_i & true_i)
    return true_positive, false_positive, false_negative

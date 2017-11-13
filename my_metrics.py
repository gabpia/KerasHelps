from keras import backend as K

# smothing factor for making the loss functions smooth
dice_smooth = 1
jaccard_smooth = 1


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    label_intersection = K.sum(y_true_f * y_pred_f)
    label_sum = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * label_intersection + dice_smooth) / (label_sum + dice_smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    label_intersection = K.sum(y_true_f * y_pred_f)
    label_union = K.sum(y_true_f) + K.sum(y_pred_f) - label_intersection
    return (label_intersection + jaccard_smooth) / (label_union + jaccard_smooth)


def jaccard_coef_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)

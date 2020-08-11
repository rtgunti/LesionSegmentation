
from tensorflow.keras import backend as K
import tensorflow as tf
from config import get_batch_size, get_nslices, get_img_size, get_patch_size
nslices = get_nslices()
batch_size = get_batch_size()
img_rows, img_cols = get_img_size()
smooth = K.epsilon()
patch_size = get_patch_size()

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def f1_score(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + smooth)
    recall = true_positives / (possible_positives + smooth)
    return 2 * ( precision * recall ) / (precision + recall + smooth)

def f1_loss(y_true, y_pred):
    return 1. - f1_score(y_true, y_pred)

def t_score(y_true, y_pred):
    cm = tf.math.confusion_matrix(y_true, tf.math.round(y_pred))

def bce_loss(y_true, y_pred):
    class_weights = [1.0, 1.0]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return -(K.max(y_pred_f,0)-y_pred_f * y_true_f + K.log(1+K.exp((-1)*K.abs(y_pred_f))))
    # return -(class_weights[1] * y_true_f * K.log(y_pred_f) + class_weights[0] * (1.0 - y_true_f) * K.log(1.0 - y_pred_f))

def wbce_loss(y_true, y_pred):
    pos_weight = 100
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(y_true_f, y_pred_f, pos_weight)
    # return -(K.max(y_pred_f,0)-y_pred_f * y_true_f + K.log(1+K.exp((-1)*K.abs(y_pred_f))))
    # return -(class_weights[1] * y_true_f * K.log(y_pred_f) + class_weights[0] * (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    return loss

def weighted_binary_crossentropy(y_true, y_pred): #Source ChoiDM
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1.-10e-8)
    loss = - (y_true * K.log(y_pred) * 0.90 + (1 - y_true) * K.log(1 - y_pred) * 0.10)
    return K.mean(loss)

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + smooth)
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + smooth)
    return recall

def tp(y_true, y_pred):
    smooth = K.epsilon()
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def tn(y_true, y_pred):
    smooth = K.epsilon()
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1.0 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 1.33
    return K.pow((1-pt_1), gamma)

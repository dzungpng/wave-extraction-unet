#!/usr/bin/env python
# https://github.com/chuckyee/cardiac-segmentation

from __future__ import division, print_function

from keras import backend as K


def soft_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    return (2 * intersection + smooth) / (area_true + area_pred + smooth)


def hard_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_sorensen_dice(y_true_int, y_pred_int, axis, smooth)


sorensen_dice = hard_sorensen_dice


def sorensen_dice_loss(y_true, y_pred, weights):
    # Input tensors have shape (batch_size, height, width, classes)
    # User must input list of weights with length equal to number of classes
    #
    # Ex: for simple binary classification, with the 0th mask
    # corresponding to the background and the 1st mask corresponding
    # to the object of interest, we set weights = [0, 1]
    batch_dice_coefs = soft_sorensen_dice(y_true, y_pred, axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * dice_coefs)


def soft_jaccard(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    union = area_true + area_pred - intersection
    return (intersection + smooth) / (union + smooth)


def hard_jaccard(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_jaccard(y_true_int, y_pred_int, axis, smooth)


jaccard = hard_jaccard


def jaccard_loss(y_true, y_pred, weights):
    batch_jaccard_coefs = soft_jaccard(y_true, y_pred, axis=[1, 2])
    jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
    w = K.constant(weights) / sum(weights)
    return 1 - K.sum(w * jaccard_coefs)


def weighted_categorical_crossentropy(y_true, y_pred, weights, epsilon=1e-8):
    ndim = K.ndim(y_pred)
    ncategory = K.int_shape(y_pred)[-1]
    # scale predictions so class probabilities of each pixel sum to 1
    y_pred /= K.sum(y_pred, axis=(ndim - 1), keepdims=True)
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    w = K.constant(weights) * (ncategory / sum(weights))
    # first, average over all axis except classes
    cross_entropies = -K.mean(y_true * K.log(y_pred), axis=tuple(range(ndim - 1)))
    return K.sum(w * cross_entropies)


def weighted_dice_coef(y_true, y_pred):
    mean = 0.21649066

    w_1 = 1 / mean ** 2
    w_0 = 1 / (1 - mean) ** 2
    y_true_f_1 = K.flatten(y_true)
    y_pred_f_1 = K.flatten(y_pred)
    y_true_f_0 = K.flatten(1 - y_true)
    y_pred_f_0 = K.flatten(1 - y_pred)

    intersection_0 = K.sum(y_true_f_0 * y_pred_f_0)
    intersection_1 = K.sum(y_true_f_1 * y_pred_f_1)

    return 2 * (w_0 * intersection_0 + w_1 * intersection_1) / (
                (w_0 * (K.sum(y_true_f_0) + K.sum(y_pred_f_0))) + (w_1 * (K.sum(y_true_f_1) + K.sum(y_pred_f_1))))


def weighted_dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - weighted_dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1 -dice_coef(y_true, y_pred)


def dice_coef_th(y_true, y_pred):
    # https://www.kaggle.com/c/ultrasound-nerve-segmentation/discussion/21358
    #intersection needs to be an array of dimension batch_size
    #the number of overlapping pixels needs to be summed over the axes for channels, rows and cols
    #intersection = K.sum(y_true * y_pred, axis=(1,2,3))
    intersection = K.sum(y_true * K.round(y_pred), axis=(3,2,1))
    smooth = 1

    #now we calculate the dice coeff for each image in the batch
    #the returned value is the mean of the dice coefficients calculated for each image

    return K.mean((2. * intersection + smooth) / (smooth + K.sum(K.round(y_true), axis=(3,2,1)) + K.sum(K.round(y_pred), axis=(3,2,1))))

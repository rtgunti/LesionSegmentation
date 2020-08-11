# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:45:16 2020
History Plots
@author: rtgun
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def plot_hist_dice_coef(history):
    plt.plot(history['dice_coef'])
    plt.plot(history['val_dice_coef'])
    plt.title('Model dice coeff')
    plt.ylabel('Dice coeff')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()

def plot_hist_f1_score(history):
    plt.plot(history['f1_score'])
    plt.plot(history['val_f1_score'])
    plt.title('Model f1_score')
    plt.ylabel('f1_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()

def plot_hist_precision(history):
    plt.plot(history['precision'])
    plt.plot(history['val_precision'])
    plt.title('Model precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()

def plot_hist_recall(history):
    plt.plot(history['recall'])
    plt.plot(history['val_recall'])
    plt.title('Model recall')
    plt.ylabel('recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()

def plot_hist_tversky(history):
    plt.plot(history['tversky'])
    plt.plot(history['val_tversky'])
    plt.title('Model Tversky Score')
    plt.ylabel('tversky')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()

def plot_hist_accuracy(history):
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train_accuracy', 'Val_accuracy'], loc='upper left')
    plt.grid()
    plt.show()

def plot_hist_loss(history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_loss', 'Val_loss'], loc='upper left')
    plt.grid()
    plt.show()


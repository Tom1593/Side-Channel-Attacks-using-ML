from time import time
import os

import keras as keras
import numpy as np
import numpy.matlib
import scipy
import random
import scipy.io as sio

from Crypto.Cipher import AES
import binascii as ba
import pickle
import multiprocess as mp
import multiprocessing  # import Pool, TimeoutError
# from joblib import Parallel, delayed, Process, Manager
# import hdf5storage
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import config
from skimage.util.shape import view_as_windows as viewW

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io import loadmat
import pandas as pd
from collections import deque
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn import metrics
import seaborn as sns
import itertools

#import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN,GRU
from keras.layers import MaxPooling1D, AveragePooling1D, Flatten, Input
from keras import backend as K
from keras.callbacks import History
from keras.optimizers import *

history = History()

# from tqdm import tqdm
from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
# import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from Information.computing import *
from Information.OnTheFly import *
from Information.MI import *
from Utils.sboxes import *
from Utils.utils import *
from Utils.OnTheFly import *
from Utils.GaussianMixture import *
from Utils.Algorithms import *
# from Kernel import KernelAttack
from Traces.LeakageOracle import *
from Traces.Preprocessing import *
from Attacks.Attacks import *
from Traces.LeakageOracle import *
from Traces.Preprocessing import *
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import hamming

num_cores = mp.cpu_count()

################################################################################################
# #`string'--> file with the opcode and exp_mode=0 where both plain and key byte change randomly for profile
# #`string_const_key'--> file with the opcode and exp_mode=2 where only plain change randomly  and key byte=0 for attack
################################################################################################

string_anat = "/OpCode_0ExpMode_1ClkDiv_3_V2/"
string = "OpCode_0ExpMode_1ClkDiv_3_V2/traces"
string_const_key = "OpCode_0ExpMode_1ClkDiv_3_V2/traces"
design = "CM"

# string_anat="/OpCode_1ExpMode_1ClkDiv_3_V2/";
# string="OpCode_1ExpMode_1ClkDiv_3_V2/traces"; design = "RCM_Dis";
# string_const_key="OpCode_1ExpMode_1ClkDiv_3_V2/traces"; design = "RCM_Dis";

# string_anat="/OpCode_8ExpMode_1ClkDiv_3_V2/";
# string="OpCode_8ExpMode_1ClkDiv_3/traces"; design = "RDR";
# string_const_key="OpCode_8ExpMode_1ClkDiv_3/traces"; design = "RDR";

# string_anat="/OpCode_19ExpMode_1ClkDiv_3_V2/";
# string="OpCode_19ExpMode_1ClkDiv_3/traces"; design = "RDR_Dis";
# string_const_key="OpCode_19ExpMode_1ClkDiv_3/traces"; design = "RDR_Dis";

save_str = "C:/Projecti/results/";

#####################################################################################################

# what to perform
predefinedPOI = 503  # # set a known POI e.g. 503
plot_final_SNR_CORR_all = 1
do_example_traces_plot = 0
plot_uniques = 0
do_SNR_etc_POI_evaluation = 0

# parameters
byte = 1  # the byte to attack
traces_start_aes = 300
traces_end_aes = 1300

########################################################


def load_file(i, str=string, chunk=''):
    dic = sio.loadmat(open(str + "_" + chunk + "%d.mat" % (i), "rb"))
    # dic = pickle.load(open(str+"_%d.mat"%(i),"rb"), encoding='latin1')
    traces = dic["traces"].astype(np.float64)
    if chunk == 'rndcomb_' or chunk == 'screen_':
        traces = traces.astype(np.float64)  # [:, 0]
    else:
        traces = traces[:, traces_start_aes:traces_end_aes].astype(np.float64)
    # # optional DC removal
    # traces = traces - np.repeat(np.mean(traces,axis=1).astype(np.float64).reshape((len(traces[:,1]), 1)), len(traces[1,:]), axis=1)
    # # optional av per file removal
    # traces = traces - np.repeat(np.mean(traces,axis=0).astype(np.float64).reshape((1, len(traces[1,:]))), len(traces[:,1]), axis=0)

    k = dic["key"].astype(np.uint8)
    p = dic["plaintext"].astype(np.uint8)
    # kp = np.zeros((256,16),dtype=np.uint8)
    # pp = np.zeros((256,16),dtype=np.uint8)
    if chunk == 'rndcomb_' or chunk == 'screen_':
        kp = np.zeros((int(2000), 16), dtype=np.uint8)  # /N_chunks),16),dtype=np.uint8)
        pp = np.zeros((int(2000), 16), dtype=np.uint8)  # /N_chunks),16),dtype=np.uint8)
    else:
        kp = np.zeros((int(2000), 16), dtype=np.uint8)
        pp = np.zeros((int(2000), 16), dtype=np.uint8)
    for i in range(16):
        kp[:, i] = k[:, i]
        pp[:, i] = p[:, i]

    return traces, kp, pp


def print_uniques(N_files=100, str=string):
    TH = np.array([0]).astype(np.int)
    TH_list = np.zeros((N_files, 1))
    for j in range(N_files):
        dic = sio.loadmat(open(str + "_%d.mat" % (j), "rb"))
        traces = dic["traces"].astype(np.float64)  # .astype(np.int16)#float64)
        TH = len(np.unique(traces))
        # print("Uniques in file %d = %d"%(int(j), int(TH)))
        TH_list[j, 0] = TH
    return TH_list


def screen_bad_files(N_files=100, THl=100, THh=4000, str=string, POI=121, traces_start_aes=traces_start_aes):
    counter = np.array([0]).astype(np.int)
    for j in range(N_files):
        dic = sio.loadmat(open(str + "_%d.mat" % (j), "rb"))
        traces = dic["traces"].astype(np.float64)  # .astype(np.int16)#float64)
        if len(np.unique(traces)) > THl or len(np.unique(traces)) < THh:
            traces = traces[:, traces_start_aes + POI - 1]
            k = dic["key"].astype(np.uint8)
            p = dic["plaintext"].astype(np.uint8)
            kp = np.zeros((20000, 16), dtype=np.uint8)
            pp = np.zeros((20000, 16), dtype=np.uint8)
            for j in range(16):
                kp[:, j] = k[:, j]
                pp[:, j] = p[:, j]

            filename = "%s_screen_%d.mat" % (str, counter)  # str + "_chunks_%d.mat"%(i*N_chunks+it)
            counter = counter + 1
            chunk = {'traces': traces.astype(np.float64), 'key': kp, 'plaintext': pp}
            sio.savemat(filename, chunk)  # {'a_dict': a_dict})
            del chunk


def get_HD(y, leng):
    y_HD = np.zeros((int(leng), ), dtype=np.uint8)
    y_HD[0] = get_HW(y[0])
    j = 1
    temp = y[0]
    bit_counts = np.array([int(bin(x).count("1")) for x in range(256)]).astype(np.uint8)
    # temp = format(y[0], '08b')
    for i in y[1:]:
        if i == leng:
            break

        y_HD[j] = np.sum(bit_counts[np.bitwise_xor(i, temp)], axis=None)
        # y_HD[j] = hamming(temp, temp1)
        j = j + 1
        temp = i
    return y_HD


def get_HD2(y, leng):
    y_HD = np.zeros((int(leng), ), order='F', dtype=np.uint8)
    #y_HD[0] = get_HW(y[0, 0])
    j = 0
    temp = 0
    bit_counts = np.array([int(bin(x).count("1")) for x in range(256)]).astype(np.uint8)
    # temp = format(y[0], '08b')
    for i in y[0, :]:
        if i == leng:
            break

        y_HD[j] = np.sum(bit_counts[np.bitwise_xor(i, temp)], axis=None)
        # y_HD[j] = hamming(temp, temp1)
        j = j + 1
        temp = i
    return y_HD

########################################################

if do_SNR_etc_POI_evaluation == 1:

    # # Plot mean traces
    i = 5
    traces, _, _ = load_file(i, str=string)
    traces = traces.astype(np.float64)
    traces2, _, _ = load_file(i, str=string)
    traces2 = traces2.astype(np.float64)

    for i in range(6):
        traces[:, :], k, p = load_file(i, str=string)

    if do_example_traces_plot == 1:
        figMean = plt.figure()
        plt.plot(np.mean(traces, axis=0))  # ,label="mean",color="b")
        plt.xlabel("time samples")
        plt.title(design)
        plt.ylabel("mean traces")
        plt.grid(True, which='both')
        # plt.legend()
        plt.show()
        figMean.savefig(save_str + design + '_meanTr' + '.png')
        figMean.savefig(save_str + design + '_meanTr' + '.pdf')
        pickle.dump(figMean, open(save_str + design + '_meanTr' + '.p', 'wb'), fix_imports=True)
        # figMeanPKL=pickle.load(open(design+'_meanTr' + '.p','rb'))
        # plt.show()
        # data=figure.axes[0].lines[0].get_data()
        # data=figure.axes[0].images[0].get_data()

    ##############################################################
    ##############################################################
    N_skip = 50

    if design == "RDR":
        N_skip = 50
        N_profile = 10000 - 1
        N_attack = 100
    if design == "RDR_Dis":
        N_skip = 50
        N_profile = 10000 - 1
        N_attack = 100
    if design == "RCM":
        N_skip = 50
        N_profile = 10000 - 1  # 1399;
        N_attack = 100
    if design == "RCM_Dis":
        N_skip = 50
        N_profile = 10000 - 1
        N_attack = 100
    if design == "CM":
        N_skip = 200
        N_profile = 4870  # 5270
        N_attack = 100

    if plot_uniques == 1:
        TH_list = print_uniques(N_files=N_profile, str=string)
        figUniq = plt.figure()
        plt.plot(TH_list)  # ,label="mean",color="b")
        plt.xlabel("filenumber")
        plt.title(design)
        plt.ylabel("Uniques")
        plt.grid(True, which='both')
        # plt.legend()
        plt.show()
        figUniq.savefig(save_str + design + '_Uniq' + '.png')
        figUniq.savefig(save_str + design + '_Uniq' + '.pdf')
        pickle.dump(figUniq, open(save_str + design + '_Uniq' + '.p', 'wb'), fix_imports=True)
        # figUniqPKL=pickle.load(open(design+'_Uniq' + '.p','rb'))
        # plt.show()
        # data=figure.axes[0].lines[0].get_data()
        # data=figure.axes[0].images[0].get_data()

    SNR_trand_tr = np.array([])
    Corr_trand_tr = np.array([])
    # POI Extraction of 1st round SBox value
    # By using a (model-less) SNR, Linear Regression or correlation with a model. In this case, we go for HW correlation
    corr_out = Corr()
    snr = SNR(256, traces_end_aes - traces_start_aes)

    l = range(N_skip, N_profile)

    iterator_rnd = list(range(N_skip, N_profile))
    random.shuffle(iterator_rnd)
    for i in tqdm(iterator_rnd):
        traces[:, :], k, p = load_file(i, str=string)  # traces,k,p = load_file(i, str=string)
        # traces = traces.astype(np.float32)
        y = np.bitwise_xor(p, k)[:, byte]  # taking the second coulom
        # model_out = get_HW(sbox_aes[y])
        snr.fit(traces, y)  # sbox_aes[y])#y)

        y_HD = get_HD(y, len(y))
        corr_out.fit(traces, y_HD)  # model_out)# y) #sbox_aes[y])

        Corr_trand_tr = np.hstack((Corr_trand_tr, np.max(np.abs(corr_out._corr))))
        SNR_trand_tr = np.hstack((SNR_trand_tr, np.max(np.abs(snr._SNR))))

    # get POI -> 1st line -from corr|| -> 2nd line -from SNR ...--> see how the RDR need many many traces to get these clear.. (i.e. SNR pick in 1-2 cycles and not on all cycles)
    POI = np.array(np.argmax(np.abs(corr_out._corr)))  # [np.argmax(np.abs(corr_out._corr)), np.argmax(np.abs(corr_in._corr))])
    POIsnr = np.array(np.argmax(np.abs(snr._SNR[:1000])))  # [np.argmax(np.abs(corr_out._corr)), np.argmax(np.abs(corr_in._corr))])
    if predefinedPOI == 1:
        POI = np.array([202])

    # Correlation with HW / Sbox output (identity)
    figCorr_HighNtr = plt.figure()
    # plt.subplot(121)
    plt.plot(corr_out._corr[:1000], label="Corr")
    plt.scatter(POI, corr_out._corr[POI], color="r", marker="2", s=200, label="POI-Corr")
    plt.scatter(POIsnr, corr_out._corr[POIsnr], color="b", marker="3", s=200, label="POI-SNR")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.xlabel("time samples")
    plt.ylabel("Corr (HW model).")
    if design == "RDR":
        plt.title("Ntr=20*10e6, " + design)
    else:
        if design == "RDR_Dis":
            plt.title("Ntr=30*10e6, " + design)
        else:
            plt.title("Ntr=10*10e6, " + design)
    plt.legend(loc='upper right')  # 'best')
    plt.grid(True, which='both')
    plt.show()
    figCorr_HighNtr.savefig(save_str + design + '_Corr_HighNtr' + '.png')
    figCorr_HighNtr.savefig(save_str + design + '_Corr_HighNtr' + '.pdf')
    pickle.dump(figCorr_HighNtr, open(save_str + design + '_Corr_HighNtr' + '.p', 'wb'), fix_imports=True)
    # figCorrPKL=pickle.load(open(design+'_Corr' + '.p','rb'))
    # plt.show()
    # data=figure.axes[0].lines[0].get_data()
    # data=figure.axes[0].images[0].get_data()

    ##Signal to Noise Ratio of SBox output
    figSNR_HighNtr = plt.figure()
    plt.plot(snr._SNR[:1000], label="SNR(y)")
    plt.scatter(POIsnr, snr._SNR[POIsnr], color="b", marker="3", s=200, label="POI-SNR")
    plt.scatter(POI, snr._SNR[POI], color="r", marker="2", s=200, label="POI-Corr")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.xlabel("time samples")
    plt.ylabel("SNR.")
    if design == "RDR":
        plt.title("Ntr=20*10e6, " + design)
    else:
        if design == "RDR_Dis":
            plt.title("Ntr=30*10e6, " + design)
        else:
            plt.title("Ntr=10*10e6, " + design)
    plt.legend(loc='lower right')  # 'best')
    plt.grid(True, which='both')
    plt.show()
    figSNR_HighNtr.savefig(save_str + design + '_SNR_HighNtr' + '.png')
    figSNR_HighNtr.savefig(save_str + design + '_SNR_HighNtr' + '.pdf')
    pickle.dump(figSNR_HighNtr, open(save_str + design + '_SNR_HighNtr' + '.p', 'wb'), fix_imports=True)
    # figSNRPKL=pickle.load(open(design+'_SNR' + '.p','rb'))
    # plt.show()
    # data=figure.axes[0].lines[0].get_data()
    # data=figure.axes[0].images[0].get_data()

    # #Correlation with HW / Sbox output (identity)
    figCorrSNRConverge_HighNtr = plt.figure()
    # plt.subplot(121)
    plt.loglog(np.arange(len(Corr_trand_tr)) * 20000, Corr_trand_tr, label="Corr(POI)")
    plt.loglog(np.arange(len(SNR_trand_tr)) * 20000, SNR_trand_tr, label="SNR(POI)")
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.xlabel("Ntr")
    plt.ylabel("Corr/SNR")
    plt.title("Corr. and SNR convergance, " + design)
    plt.legend(loc='upper right')  # 'best')
    plt.grid(True, which='both')
    plt.show()
    figCorrSNRConverge_HighNtr.savefig(save_str + design + '_CorrSNRConverge_HighNtr' + '.png')
    figCorrSNRConverge_HighNtr.savefig(save_str + design + '_CorrSNRConverge_HighNtr' + '.pdf')
    pickle.dump(figCorrSNRConverge_HighNtr, open(save_str + design + '_CorrSNRConverge_HighNtr' + '.p', 'wb'),
                fix_imports=True)
    # figCorrPKL=pickle.load(open(design+'_Corr' + '.p','rb'))
    # plt.show()
    # data=figure.axes[0].lines[0].get_data()
    # data=figure.axes[0].images[0].get_data()

    ################################################################
    ############## FINAL ALL SNR CURVES LOGLOG #####################
    ################################################################
    if plot_final_SNR_CORR_all == 1:
        figtmp = pickle.load(open('C:/Projecti/results/CM_CorrSNRConverge_HighNtr.p', 'rb'))
        CM1 = figtmp.axes[0].lines[0].get_data()[1]
        CM2 = figtmp.axes[0].lines[1].get_data()[1]
        # figtmp = pickle.load(open('RCM_CorrSNRConverge_HighNtr.p', 'rb'))
        # RCM1 = figtmp.axes[0].lines[0].get_data()[1]
        # RCM2 = figtmp.axes[0].lines[1].get_data()[1]
        # figtmp = pickle.load(open('RCM_Dis_CorrSNRConverge_HighNtr.p', 'rb'))
        # RCMdis1 = figtmp.axes[0].lines[0].get_data()[1]
        # RCMdis2 = figtmp.axes[0].lines[1].get_data()[1]
        # figtmp = pickle.load(open('RDR_CorrSNRConverge_HighNtr.p', 'rb'))
        # RDR1 = figtmp.axes[0].lines[0].get_data()[1]
        # RDR2 = figtmp.axes[0].lines[1].get_data()[1]
        # figtmp = pickle.load(open('RDR_Dis_CorrSNRConverge_HighNtr.p', 'rb'))
        # RDRdis1 = figtmp.axes[0].lines[0].get_data()[1]
        # RDRdis2 = figtmp.axes[0].lines[1].get_data()[1]

        figSNRConverge_HighNtr_ALL = plt.figure()
        plt.loglog(np.arange(len(CM2)) * 20000, CM2, label="CMOS")
        # plt.loglog(np.arange(len(RCM2)) * 20000, RCM2, label="R-CMOS, disabled")
        # plt.loglog(np.arange(len(RCMdis2)) * 20000, RCMdis2, label="R-CMOS")
        # plt.loglog(np.arange(len(RDR2)) * 20000, RDR2, label="R-DR, disabled")
        # plt.loglog(np.arange(len(RDRdis2)) * 20000, RDRdis2, label="R-DR")

        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        plt.xlabel('Ntr')
        plt.ylabel("SNR(POI)")
        plt.title("SNR convergance - rolled and fully-parallel AES.")
        plt.legend(loc='lower left')  # 'best')
        plt.grid(True, which='both')
        plt.ylim((0.000008, 0.02))
        plt.annotate('Decoupling', xy=(2e6, 1e-3), xytext=(3e5, 2e-3),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.annotate('Randomizer', xy=(4e6, 2e-4), xytext=(8e5, 6e-4),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.annotate('', xy=(8e6, 5e-5), xytext=(3.5e6, 1.31e-4),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.annotate('Signal-Red.', xy=(1e6, 1e-4))
        # plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
        #        arrowprops=dict(facecolor='black', shrink=0.05))

        plt.show()
        figSNRConverge_HighNtr_ALL.savefig('SNRConverge_HighNtr_ALL' + '.png')
        figSNRConverge_HighNtr_ALL.savefig('SNRConverge_HighNtr_ALL' + '.pdf')
        pickle.dump(figSNRConverge_HighNtr_ALL, open('SNRConverge_HighNtr_ALL' + '.p', 'wb'), fix_imports=True)

        figCORRConverge_HighNtr_ALL = plt.figure()
        plt.loglog(np.arange(len(CM1)) * 20000, CM1, label="CMOS")
        # plt.loglog(np.arange(len(RCM1)) * 20000, RCM1, label="R-CMOS, disabled")
        # plt.loglog(np.arange(len(RCMdis1)) * 20000, RCMdis1, label="R-CMOS")
        # plt.loglog(np.arange(len(RDR1)) * 20000, RDR1, label="R-DR, disabled")
        # plt.loglog(np.arange(len(RDRdis1)) * 20000, RDRdis1, label="R-DR")

        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        plt.xlabel("Ntr")
        plt.ylabel("Correlation(POI)")
        plt.title("Corr(HW) convergance - rolled and fully-parallel AES.")
        plt.legend(loc='lower left')  # 'best')
        plt.grid(True, which='both')

        plt.annotate('Decoupling', xy=(1e6, 1e-2), xytext=(1e5, 2e-2),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.annotate('Randomizer', xy=(3e6, 2.5e-3), xytext=(7e5, 6e-3),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.annotate('Signal-Red.', xy=(7e6, 1.5e-3), xytext=(1e6, 2.5e-3),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        # plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
        #        arrowprops=dict(facecolor='black', shrink=0.05))

        plt.show()
        figCORRConverge_HighNtr_ALL.savefig(save_str + 'CORRConverge_HighNtr_ALL' + '.png')
        figCORRConverge_HighNtr_ALL.savefig(save_str + 'CORRConverge_HighNtr_ALL' + '.pdf')
        pickle.dump(figCORRConverge_HighNtr_ALL, open(save_str + 'CORRConverge_HighNtr_ALL' + '.p', 'wb'),
                    fix_imports=True)

########################################################
N_skip = 50

if design == "RDR":
    N_skip = 50
    N_profile = 10000 - 1
    N_attack = 100
    POIsnr = 250
if design == "RDR_Dis":
    N_skip = 50
    N_profile = 10000 - 1
    N_attack = 100
if design == "RCM":
    N_skip = 50
    N_profile = 10000 - 1  # 1399;
    N_attack = 100
if design == "RCM_Dis":
    N_skip = 50
    N_profile = 10000 - 1
    N_attack = 100
if design == "CM":
    N_skip = 200
    N_profile = 4870  # 5270
    N_attack = 100
    POIsnr = 81  # for the CM design

Final_POI = POIsnr + 300

# directory=os.path.dirname(os.path.realpath(__file__)) + string_anat #'/OpCode_8ExpMode_1ClkDiv_3_V2/'
directory = "C:/Projecti" + string_anat
RangeOrPOI = 1  # 0=selected POI, 1=a range of set_size/2 from both POI sides
poi_dict = {'traces': Final_POI, 'plaintext': 1, 'key': 1}
sample_size = 500  # 2400#1000#2800 #0.8*N_profile #4870 #2800 ## Number of files
# num_of_files = 3055
mode = 'y_xor'
# agg_limit=9
BatchOrAvg = 1  # 0=Avg (set_size !=1), 1=Both (set_size!=1), 2=Batch (any set size) [- need to add]
set_size = 500  # 2000 # size of batch (repetitions of leakage in a  U-V case)
Nfeatures = 50
feat_select = 0
Do_balance = 0  # use SMOTE to augmant the imballanced classes  - when e.g. grouping to imballanced sets of HW
Do_HD = 1  # 9 classes versus 256 ...


def open_sheet(filename, sheet):
    matfile = loadmat(filename)
    data = matfile[sheet]
    if RangeOrPOI == 1 and sheet == "traces":
        column_poi = data[:, int(poi_dict[sheet] - 1 - 1 - set_size / 2): int(poi_dict[sheet] - 1 + 1 + set_size / 2)]
    else:
        column_poi = data[:, poi_dict[sheet] - 1]
    # print(column_poi.shape)
    return column_poi


def get_file_index(filename):
    if filename.startswith("traces_"):
        ind_start = filename.index("_") + 1
        index = filename[ind_start:len(filename)]
    else:
        index = 0
    return int(index)


def create_dataset(x, y):
    return np.column_stack((x, y))


def counts_per_key(data):
    unique, counts = np.unique(data[:, 1], return_counts=True)
    ans = dict(zip(unique, counts))
    # print(ans)


def create_df(data):
    index = ['Row' + str(i) for i in range(1, len(data) + 1)]
    df = pd.DataFrame(data=data, index=index, columns=["x", "y"])
    return df


def calc_avg(data, agg_size):
    df = create_df(data)
    #    index = ['Row'+str(i) for i in range(1, len(data)+1)]
    #    df = pd.DataFrame(data=data, index=index, columns=["x", "y"])
    avg_matrix = pd.DataFrame(columns=['y', 'size', 'avg_x'])
    if Do_HD == 1:
        r = 8
    else:
        r = 255
    for i in range(r):
        for j in range(agg_limit):
            agg_size = pow(2, j + 1)  # power
            group = df[df['y'] == i]['x'].head(agg_size)
            res = float(sum(group) / len(group))
            avg_matrix = avg_matrix.append({'y': i, 'size': agg_size, 'avg_x': res}, ignore_index=True)
            # print(avg_matrix)


def create_set(data):
    groups = dict()
    df = create_df(data).sort_values(by=['y'])  # df sorted by y values
    final_set = []
    current = deque(maxlen=set_size + 2)  # creating the stack constract which will contain all points + the AVG
    current_avg = deque(maxlen=2)  # creating the stack constract which will contain only the AVG
    last_y = df['y'][0]
    for value in tqdm(df.values, desc="create_set"):
        current_y = value[1]
        if current_y == last_y:
            current.append(value[0])  # appeding set_size leakages (per y)
        else:
            current = deque(maxlen=set_size + 2)
        if (len(current) == set_size):
            avg = sum(current) / len(current)
            current.append(avg)  # appeding averaged leakage
            current.append(current_y)  # Y lable is appended at the end
            current_avg.append(avg)
            current_avg.append(current_y)
            final_set.append(current)  # append new sample line for whole the repentitions
            # final_set.append(current_avg) #append new sample line for only the average
            current = deque(maxlen=set_size + 2)  # empty temp sequence
            current_avg = deque(maxlen=2)
        last_y = current_y
    return final_set


def prep_features(data):
    res = []
    for each in tqdm(data, desc="prep_features"):
        if BatchOrAvg == 1:
            res.append(list(each)[:-1])  # [0:set_size+1]) #all the feature elements but the lable
        else:
            res.append(list(each)[0])
    return (res)


def load_data(sample_size):
    # for i, filename in tqdm(enumerate(os.listdir(directory)), desc="load_data %d of %d" % (i, sample_size)):
    for i in range(sample_size):
        filename = "traces_"+str(i)
        index = get_file_index(filename)
        if index > 50:  # filter first 50 traces
            trace = open_sheet(directory + filename, 'traces')
            plaintext = open_sheet(directory + filename, 'plaintext')

            x_temp = trace
            y_temp = open_sheet(directory + filename, 'key')
            if Do_HD == 1:
                xor = np.bitwise_xor(np.array(plaintext).astype(np.uint8).reshape(len(np.array(plaintext)), 1),
                                     np.array(y_temp).astype(np.uint8).reshape(len(np.array(y_temp)), 1))
                y_xor_temp = get_HD(xor, len(xor))
                # print(y_xor_temp)
            else:
                y_xor_temp = np.bitwise_xor(plaintext, y_temp)
            if i <= 0.8 * sample_size:
                # add to sample
                if 'x' not in locals():
                    x = x_temp
                else:
                    x = np.append(x, x_temp, axis=0)
                    # print((x.shape))
                if 'y' not in locals():
                    y = y_temp
                    y_xor = y_xor_temp
                else:
                    y = np.append(y, y_temp)
                    y_xor = np.append(y_xor, y_xor_temp)
            else:
                # prepare test
                if 'x_test' not in locals():
                    x_test = x_temp
                else:
                    x_test = np.append(x_test, x_temp, axis=0)
                if 'y_test' not in locals():
                    y_test = y_temp
                else:
                    y_test = np.append(y_test, y_temp, axis=0)
                if 'y_xor_test' not in locals():
                    y_xor_test = y_xor_temp
                else:
                    y_xor_test = np.append(y_xor_test, y_xor_temp, axis=0)
            if i > sample_size:
                break
    if mode == 'y_xor':
        return x, y_xor, x_test, y_xor_test
    elif mode == 'y':
        return x, y, x_test, y_test


def prep_y(data):
    res = []
    for each in tqdm(data, desc="prep_y"):
        if BatchOrAvg == 1:
            res.append(list(each)[-1])  # [set_size+1]) #The last lable element
        else:
            res.append(list(each)[1])
    return res


def load_data_all(sample_size):
    # for i, filename in tqdm(enumerate(os.listdir(directory)), desc="load_data %d of %d" % (i, sample_size)):
    for i in range(sample_size):
        filename = "traces_"+str(i)
        index = get_file_index(filename)
        if index > 50:  # filter first 50 traces
            trace = open_sheet(directory + filename, 'traces')

            plaintext = open_sheet(directory + filename, 'plaintext')

            x_temp = trace
            y_temp = open_sheet(directory + filename, 'key')
            if Do_HD == 1:
                xor = np.bitwise_xor(np.array(plaintext).astype(np.uint8).reshape(len(np.array(plaintext)), 1),
                                     np.array(y_temp).astype(np.uint8).reshape(len(np.array(y_temp)), 1))
                # y_xor_temp = np.sum(np.unpackbits(xor, axis=1), axis=1)
                y_xor_temp = get_HD(xor, len(xor))
            else:
                y_xor_temp = np.bitwise_xor(plaintext, y_temp)
            # add to sample
            if 'x' not in locals():
                x = x_temp
            else:
                x = np.append(x, x_temp, axis=0)
                # print((x.shape))
            if 'y' not in locals():
                y = y_temp
                y_xor = y_xor_temp
            else:
                y = np.append(y, y_temp)
                y_xor = np.append(y_xor, y_xor_temp)
            if i > sample_size:
                break
    if mode == 'y_xor':
        return x, y_xor
    elif mode == 'y':
        return x, y


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, index='design_name'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    color_CM = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    color_CM.savefig(save_str + 'Color_confMat' + index + '.png')
    color_CM.savefig(save_str + 'Color_confMat' + index + '.pdf')
    pickle.dump(color_CM, open(save_str + 'Color_confMat' + index + '.p', 'wb'), fix_imports=True)


def exists(path):
    try:  # is needed so it won't try to actually open the file unless needed, not to lock it, corrupt it ... etc.
        st = os.stat(path)
    except os.error:
        return False
    return True


# def main():

###############################################################################
# ############## Data Preperation ##############################################
DB_str = save_str + "DataSet_DesignOp8_Nfiles_%d_feat_%d_reduced_%d_file%d.mat" % (sample_size, set_size, Nfeatures, 0)
if os.path.exists(DB_str):
    Nfiles_store = 10
    x = []
    y = []
    for j in range(Nfiles_store):
        DB_str_sv = save_str + "DataSet_DesignOp8_Nfiles_%d_feat_%d_reduced_%d_file" % (
        sample_size, set_size, Nfeatures) + "%d.mat" % (j)
        dic = sio.loadmat(open(DB_str_sv, "rb"))
        if len(x) == 0:
            x = dic["x"].astype(np.int16)
            y = dic["y"].astype(np.uint64)
            y = np.squeeze(y)
        else:
            x = np.vstack((x, dic["x"].astype(np.int16)))
            y = np.hstack((y, np.squeeze(dic["y"].astype(np.uint64))))
            y = np.squeeze(y)
else:
    x, y = load_data_all(sample_size)  # load all data set
    Nfiles_store = 10
    for j in range(Nfiles_store):
        DB_str_sv = save_str + "DataSet_DesignOp8_Nfiles_%d_feat_%d_reduced_%d_file" % (
        sample_size, set_size, Nfeatures) + "%d.mat" % (j)
        xt = x[1 + j * (int(len(x) / Nfiles_store)):1 + (j + 1) * (int(len(x) / Nfiles_store)), :]
        yt = y[1 + j * (int(len(y) / Nfiles_store)):1 + (j + 1) * (int(len(y) / Nfiles_store))]
        chunk = {'x': xt.astype(np.int16), 'y': yt.astype(np.uint64)}
        sio.savemat(DB_str_sv, chunk)  # {'a_dict': a_dict})

scaler = MinMaxScaler()  # create a "scaler" object to scale leakages to 0-1 range for feature ext.
scaler.fit(x)
x = scaler.transform(x)
global x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  # train,test splits
sns.countplot(y_train)  # plot our HD distribution of the sensitive var ...

if feat_select == 1:
    fselect = SelectKBest(chi2, k=Nfeatures)  # take Nfeatures best features with pearson chi2 test (needs non negative vals)
    x_train = fselect.fit_transform(x_train, y_train)  # take the training features
    x_test = fselect.transform(x_test)  # take the testing features
    features_list = fselect.get_support()
    feature_indexes = [i for i, x in enumerate(features_list) if x]

# Add information theoratic feature selection 
# Add multivariate feature selection
    Feat_select = plt.figure()
    plt.plot(np.mean(x, axis=0))  # ,label="mean",color="b")
    plt.xlabel("time samples")
    plt.ylabel("mean tr./ Features selection")
    plt.grid(True, which='both')
    plt.scatter(feature_indexes, np.mean(x, axis=0)[feature_indexes], color='black')
    plt.title('Feature Selection ' + design)
    plt.legend(['Avg. trace', 'feat.'], loc='upper right')
    plt.show()
    Feat_select.savefig(save_str + 'Feat_select' + '.png')
    Feat_select.savefig(save_str + 'Feat_select' + '.pdf')
    pickle.dump(Feat_select, open(save_str + 'Feat_select' + '.p', 'wb'), fix_imports=True)

###############################################################################
# #### Section for imballanced classes - not the case for identity classes - 2^8
# y_final=np.array(y_final).reshape(len(np.array(y_final)),1)
print('y size before SMOTE:', len(y_train))
if Do_balance == 1:
    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    print('y size after SMOTE:', len(y_train))
"""""
# Filter requirements.

fs = 30.0       # sample rate, Hz
cutoff = 2      # desired cutoff frequency of the filter, Hz
order = 6       # sin wave can be approx represented as quadratic

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


traceplot = plt.figure()
plt.plot(x[0], color="black")
x[0] = butter_lowpass_filter(x[0], cutoff, fs, order)
plt.plot(x[0], color="green")

"""
###############################################################################
############### LSTM / RNN ####################################################
# ==> For now it completely missed the "time relation within the leakage trace  in the sense that 1) RNN structure should make use of it.
# ==> 2) the feedback structure can be played with with shorter /longer feedback paths [ for now each layer has one layer feedback and thats it]
# ==> Remove the feature reduction - shape in two "time sequences" for the RNN proparties etc ... with time samples ...
# ==> and add ", return_sequences=True"
# ==> OR - take features as "time instances" ... transpose ..

# Should we fut all dims here of the leakage and let it do its own feature selection ?

def strided_method(ar):
    a = np.concatenate((ar, ar[:-1]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L - 1:], (L, L), (-n, n))


def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    a_ext = np.concatenate((a, a[:, :-1]), axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext, (1, n))[np.arange(len(r)), (n - r) % n, 0]


max_features = 502  # 1400
maxlen = Nfeatures  # 50
nb_classes = 9  # starts from 0

# transform  to sequences and pad them / truncate ...
y_train_cat = np_utils.to_categorical(y_train, nb_classes)
y_test_cat = np_utils.to_categorical(y_test, nb_classes)
batch_size = 500

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

print('Pad sequences (samples x time)')
x_train_sequ = sequence.pad_sequences(x_train, maxlen=maxlen + 5, dtype=float)  # 30 features now 25+5
x_test_sequ = sequence.pad_sequences(x_test, maxlen=maxlen + 5, dtype=float)
# test = np.linspace(0, 29, 30)
# x_train_sequ = np.array([test, list(np.array(test)+1)])
# x_test_sequ  = np.array([test,  list(np.array(test)+1), list(np.array(test)+2)])

shifts_amnt = 5  # adding five replicas of each row
x_train_sequ_resh = np.repeat(x_train_sequ, repeats=shifts_amnt + 1, axis=0)
x_test_sequ_resh = np.repeat(x_test_sequ, repeats=shifts_amnt + 1, axis=0)

# x_train_sequ_resh = np.reshape(x_train_sequ_resh, (len(x_train_sequ)*(shifts_amnt+1) , (maxlen+5) ) )
# x_test_sequ_resh  = np.reshape(x_test_sequ_resh, (len(x_test_sequ)*(shifts_amnt+1) , (maxlen+5) ) )

shifts_once = np.linspace(0, 5, shifts_amnt + 1)  # shifting each row in 0, 1, ..., 5 in a loop... 0, 1, ...5, 1, 1, ...5, 0, 1,...
shifts_multiple = np.repeat(np.expand_dims(shifts_once, axis=1), int(len(x_train_sequ_resh) / (shifts_amnt + 1)), axis=1)
shifts = np.reshape(shifts_multiple.T, (1, int(len(shifts_multiple) * len(shifts_multiple.T))))

x_train_sequ_resh = strided_indexing_roll(x_train_sequ_resh, shifts.astype(int))
x_train_sequ_resh = np.squeeze(x_train_sequ_resh, axis=0)
# x_train_sequ_resh = np.reshape(x_train_sequ_resh, (int(len(x_train_sequ_resh)), maxlen + 5))
x_train_sequ_resh = np.reshape(x_train_sequ_resh, (int(len(x_train_sequ_resh) / (shifts_amnt+1)), (shifts_amnt+1), maxlen+5))


shifts_multiple = np.repeat(np.expand_dims(shifts_once, axis=1), int(len(x_test_sequ_resh) / (shifts_amnt + 1)), axis=1)
shifts = np.reshape(shifts_multiple.T, (1, int(len(shifts_multiple) * len(shifts_multiple.T))))

x_test_sequ_resh = strided_indexing_roll(x_test_sequ_resh, shifts.astype(int))
x_test_sequ_resh = np.squeeze(x_test_sequ_resh, axis=0)
# x_test_sequ_resh = np.reshape(x_test_sequ_resh, (int(len(x_test_sequ_resh)), maxlen + 5))
x_test_sequ_resh = np.reshape(x_test_sequ_resh, (int(len(x_test_sequ_resh) / (shifts_amnt+1)), (shifts_amnt+1), maxlen+5))


# x_train_sequ_resh = x_train_sequ_resh.reshape(, 505)
# x_test_sequ_resh = x_test_sequ.reshape((len(x_test), 3, 10))

# we already have catagorical lables nb_classes=9

# y_train_x5 = np.repeat(y_train_cat, repeats=shifts_amnt + 1, axis=0)
# y_test_x5 = np.repeat(y_test_cat, repeats=shifts_amnt + 1, axis=0)
print('X_train shape:', x_train_sequ.shape)
print('X_test shape:', x_test_sequ.shape)

# #####################LSTM######################
print('Build model...')
"""""
model = Sequential()
# model.add(Embedding(input_length=maxlen + shifts_amnt, input_dim=len(np.unique(x)), output_dim=9))  # The embedding layer is really important: a way to map our input into 128 dimension space here. The layer is trained through iterations (epochs) to have a better weights for the features that allow to minimize the global error of the network.
# model.add(Dropout(0.2))
model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, activation='relu', return_sequences=True))  # return_sequences=True)) #input_shape=(shifts_amnt + 1, int((maxlen + 5)))
# model.add(
#    LSTM(64,dropout=0.2,recurrent_dropout=0.2,activation='relu', input_shape=(shifts_amnt + 1, int((maxlen + 5))),
#   return_sequences=True))  # ,  input_shape =(1, maxlen))) # one LSTM layer with 128 neurons #return_sequences=False,

model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1, activation='relu'))  # , return_sequences=True# one LSTM layer with 128 neurons
# model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.1, activation='relu', return_sequences=True))#, return_sequences=True))  # one LSTM layer with 128 neurons

# model.add(LSTM(64, recurrent_dropout=0.2,dropout=0.2,activation='selu', ))  # , return_sequences=True))  # one LSTM layer with 128 neurons

# model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.1, activation='relu', return_sequences=True))#, return_sequences=True))  # one LSTM layer with 128 neurons
# model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.1, activation='relu', return_sequences=True))  # 1
# model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.1, activation='relu', return_sequences=True))  # 2
# model.add(Dense(64, activation='relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
"""
model = Sequential()
# model.add(Embedding(input_length = maxlen, input_dim = len(np.unique(x)), output_dim = 256)) # The embedding layer is really important: a way to map our input into 128 dimension space here. The layer is trained through iterations (epochs) to have a better weights for the features that allow to minimize the global error of the network.
# model.add(Dropout(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.1, activation='relu',   input_shape=(shifts_amnt+1, int(maxlen+5)), return_sequences=True))  #,  input_shape =(1, maxlen))) # one LSTM layer with 128 neurons #return_sequences=False,
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.1, activation='relu'))  # , return_sequences=True))  # LSTM layer with 64 neurons
# model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.1, activation='relu'))  # LSTM layer with 32 neurons
# Dropout for regularization
# model.add(SimpleRNN(32))  # , activation='softmax'))
model.add(Dense(nb_classes))   # Fully connected layer
model.add(Activation('softmax'))

optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # 'adam'
model.summary()

print('Train...')
# print(len(x_train_sequ_resh))
# print(len(y_train))
history = model.fit(x_train_sequ_resh, y_train_cat, batch_size=int(np.floor(((1/10)*batch_size))), epochs=100,
                    validation_data=(x_test_sequ_resh, y_test_cat), class_weight=class_weights)  # should I train in BATCHES ? "batch_size=batch_size,"
score, acc = model.evaluate(x_test_sequ_resh, y_test_cat, batch_size=10 * batch_size)
# history = model.fit(x_train_sequ.reshape((len(x_train_sequ), len(x_train_sequ.T), 1)), y_train_cat,   batch_size=batch_size,   epochs=100,   validation_data=(x_test_sequ.reshape((len(x_test_sequ), len(x_test_sequ.T), 1)), y_test_cat), class_weight=class_weights) # should I train in BATCHES ? "batch_size=batch_size,"
# score, acc = model.evaluate(x_test_sequ.reshape((len(x_test_sequ), len(x_test_sequ.T), 1)), y_test_cat,  batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print(history.history.keys())
Accplot = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
Accplot.savefig(save_str + 'RNN_ACC' + '.png')
Accplot.savefig(save_str + 'RNN_ACC' + '.pdf')
pickle.dump(Accplot, open(save_str + 'RNN_ACC' + '.p', 'wb'), fix_imports=True)

Lossplot = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
Lossplot.savefig(save_str + 'RNN_Loss' + '.png')
Lossplot.savefig(save_str + 'RNN_Loss' + '.pdf')
pickle.dump(Lossplot, open(save_str + 'RNN_Loss' + '.p', 'wb'), fix_imports=True)

print("Generating test predictions...")
global y_predict4
y_predict4 = np.argmax(model.predict(x_test_sequ_resh), axis=-1)  # ape((len(x_test_sequ), len(x_test_sequ.T), 1))), axis=-1) #recommanded for softmax last layer activation#model.predict_classes(x_test, verbose=0)
y_predict4 = y_predict4.astype(int)
# plot_tree_results_xor_bits(y_predict4,y_test, index='RNN')
# print('prediction accuracy: ', accuracy_score(y_test, y_predict4+1))
print(confusion_matrix(y_test, y_predict4))
print(classification_report(y_test, y_predict4, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8']))
print(accuracy_score(y_predict4, y_test))
cnf_matrix4 = confusion_matrix(y_test, y_predict4)
plot_confusion_matrix(cnf_matrix4, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8'],
                      title='Confusion matrix, without normalization', index='RNN')

global y_predict4_chunk


def Rank_keys_RNN(keydelta):
    y_test_delta = np.bitwise_xor(np.matlib.repmat(np.array(keydelta).astype(np.uint8), 1, len(np.array(y_test_chunk))),
                                  np.array(y_test_chunk).astype(np.uint8))

    # upby = np.unpackbits(y_test_delta, axis=0)
    # HWy = np.sum(upby, axis=0)

    HD_y = get_HD2(y_test_delta, np.size(y_test_delta, 1))

    # HD_y = np.zeros((1, int(len(y_test_delta))), dtype=np.uint8)
    # j = 0
    # for i in HDy:
    #     HD_y[j] = i
    #     j = j+1

    cm = confusion_matrix(np.array(y_predict4_chunk), np.array(HD_y))
    predictions_acc = np.sum(np.diag(cm)) / np.sum(cm)
    return predictions_acc


print("predict_iterations_Ranking_RNN")
steps = 20
rank_RNN = np.zeros(steps)
MSE_RNN = np.zeros(steps)
acc_RNN = np.zeros(steps)
for j in tqdm((range(1, steps + 1)), desc="load_data %d of %d" % (j, steps)):
    y_predict4_chunk = y_predict4[1:(j - 1) * 20 + 2]  # 2**(j-1)+1] #int(len(y_predict4.T)/2**(j-1))]
    y_test_chunk = y_test[1:(j - 1) * 20 + 2]  # 2**(j-1)+1] #int(len(y_predict4.T)/2**(j-1))]

    finals = []
    results = []
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(Rank_keys_RNN, np.linspace(0, 255, 256))
        for value in results:
            finals.append(value)
        # print(results.result())
    acc_RNN[j - 1] = np.array(finals)[0]
    rank_RNN[j - 1] = 255 - np.argwhere(np.array(np.argsort(np.array(finals))) == 0)
    MSE_RNN[j - 1] = np.sqrt(metrics.mean_squared_error(y_predict4_chunk, y_test_chunk))
    print("Rank (Guessing Entropy),%d/%d test:" % (j, steps), rank_RNN[j - 1])
    print("Root Mean Squared Error (RMSE), %d/%d test:" % (j, steps), MSE_RNN[j - 1])
    print("Acc,%d/%d test:" % (j, steps), acc_RNN[j - 1])

ind = np.zeros(steps)
j = range(1, steps + 1)
for j in (range(1, steps + 1)):
    ind[j - 1] = (j - 1) * 20 + 1  # 2**(j-1) #int(len(y_predict4.T)/2**(j-1))
GE_RNN = plt.figure()
# ax = GE_RF.gca()
plt.plot(ind, rank_RNN, color='black')
plt.title('Rank (Partial Guessing Entropy)')
plt.ylabel('PGE')
plt.xlabel('Testing set size (N traces)')
# GE_RF.legend(['train','test'], loc='upper left')
plt.show()
GE_RNN.savefig(save_str + 'RNN_PGE' + '.png')
GE_RNN.savefig(save_str + 'RNN_PGE' + '.pdf')
pickle.dump(GE_RNN, open(save_str + 'RNN_PGE' + '.p', 'wb'), fix_imports=True)

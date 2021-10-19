import header
import importlib

importlib.reload(header)  # For reloading after making changes
from header import *


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier


USE_THREE_BOTS = 0

DEBUG = 1

network_traffic = 1

CAT_FEATURES = 0

CTU_NERIS = 0
USE_ONE_BOT = 1
USE_ALL_BOTS = 0
USE_FOUR_BOTS = 0

REMOVE_SKEW = 0

USE_FEATURE_REDUCTION = 0


def top_features(data, classifier=""):

    X = data.drop(["Label"], axis=1).values
    y = data["Label"].values

    clf = XGBClassifier()

    # if(classifier=='XGB'):
    print("Evaluation ---->> XBG")
    clf = XGBClassifier()

    # if ALL_CLASSIFIERS:
    #     if (classifier=='DT'):
    #         print('Evaluation ---->> DT')
    #         clf = DecisionTreeClassifier()
    #     elif (classifier=='RF'):
    #         print('Evaluation ---->> RF')
    #         clf=RandomForestClassifier(n_estimators=100)

    clf.fit(X, y)

    importance = clf.feature_importances_

    importance = [importance.tolist()]
    print(importance)

    important_features_dict = {}

    # important_list = list(map(lambda el:[el], importance))

    dictlist = []

    print(importance)

    importance_df = pd.DataFrame(
        importance, columns=data.drop(["Label"], axis=1).columns
    )

    print(importance_df)

    for x, i in enumerate(importance_df.columns):
        important_features_dict[x] = i

    important_features_list = sorted(
        important_features_dict, key=important_features_dict.get, reverse=True
    )

    print("Most important features: %s" % important_features_list)

    importance_df = importance_df.loc[:, (importance_df != 0).any(axis=0)]

    print(len(importance_df.columns))

    return importance_df.columns

    # # # summarize feature importance
    # # for i,v in enumerate(importance):
    # #     print('Feature: %0d, Score: %.5f' % (i,v))
    # # # plot feature importance
    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()


def prepare_ISCX_2014_data(PATH="", INPUT_FILE_NAME=""):

    # =============================================================================================
    # Read Data
    # =============================================================================================
    data = pd.read_csv(str(PATH) + str(INPUT_FILE_NAME), low_memory=False)

    print(
        "Processing File: " + str(INPUT_FILE_NAME) + " DATA shape: " + str(data.shape)
    )

    # =============================================================================================
    # Drop all categorical features
    # =============================================================================================

    if network_traffic:
        data = data.drop(
            ["Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Protocol"],
            axis=1,
        ).copy()
        if DEBUG:
            print("data_df after removing categorical features")
            print(data.shape)

        # =============================================================================================
        # Choose only Flow ID and Label columns for labeling purposes in another data frame
        # =============================================================================================

        data_df = data[["Flow ID", "Label"]].copy()

        # =============================================================================================
        # remove any columns with all values = Zero
        # =============================================================================================

        data = data.loc[:, (data != 0).any(axis=0)]

        # IRC = data_df['Flow ID'].str.contains('192.168.2.112-131.202.243.84|192.168.5.122-198.164.30.2|192.168.2.110-192.168.5.122|192.168.4.118-192.168.5.122|192.168.2.113-192.168.5.122|192.168.1.103-192.168.5.122|192.168.4.120-192.168.5.122|192.168.2.112-192.168.2.110|192.168.2.112-192.168.4.120|192.168.2.112-192.168.1.103|192.168.2.112-192.168.2.113|192.168.2.112-192.168.4.118|192.168.2.112-192.168.2.109|192.168.2.112-192.168.2.105|192.168.1.105-192.168.5.122')

        IRC_1 = data_df["Flow ID"].str.contains("192.168.2.112-131.202.243.84")
        IRC_2 = data_df["Flow ID"].str.contains("192.168.5.122-198.164.30.2")
        IRC_3 = data_df["Flow ID"].str.contains("192.168.2.110-192.168.5.122")
        IRC_4 = data_df["Flow ID"].str.contains("192.168.4.118-192.168.5.122")
        IRC_5 = data_df["Flow ID"].str.contains("192.168.2.113-192.168.5.122")
        IRC_6 = data_df["Flow ID"].str.contains("192.168.1.103-192.168.5.122")
        IRC_7 = data_df["Flow ID"].str.contains("192.168.4.120-192.168.5.122")
        IRC_8 = data_df["Flow ID"].str.contains("192.168.2.112-192.168.2.110")
        IRC_9 = data_df["Flow ID"].str.contains("192.168.2.112-192.168.4.120")
        IRC_10 = data_df["Flow ID"].str.contains("192.168.2.112-192.168.1.103")
        IRC_11 = data_df["Flow ID"].str.contains("192.168.2.112-192.168.2.113")
        IRC_12 = data_df["Flow ID"].str.contains("192.168.2.112-192.168.4.118")
        IRC_13 = data_df["Flow ID"].str.contains("192.168.2.112-192.168.2.109")
        IRC_14 = data_df["Flow ID"].str.contains("192.168.2.112-192.168.2.105")
        IRC_15 = data_df["Flow ID"].str.contains("192.168.1.105-192.168.5.122")

        Neris = data_df["Flow ID"].str.contains("147.32.84.180")
        RBot = data_df["Flow ID"].str.contains("147.32.84.170")
        Menti = data_df["Flow ID"].str.contains("147.32.84.150")
        Sogou = data_df["Flow ID"].str.contains("147.32.84.140")
        Murlo = data_df["Flow ID"].str.contains("147.32.84.130")
        Virut = data_df["Flow ID"].str.contains("147.32.84.160")
        Black_hole_1 = data_df["Flow ID"].str.contains("10.0.2.15")
        Black_hole_2 = data_df["Flow ID"].str.contains("192.168.106.141")
        Black_hole_3 = data_df["Flow ID"].str.contains("192.168.106.131")
        TBot_1 = data_df["Flow ID"].str.contains("172.16.253.130")
        TBot_2 = data_df["Flow ID"].str.contains("172.16.253.131")
        TBot_3 = data_df["Flow ID"].str.contains("172.16.253.129")
        TBot_4 = data_df["Flow ID"].str.contains("172.16.253.240")
        Weasel_master = data_df["Flow ID"].str.contains("74.78.117.238")
        Weasel_bot = data_df["Flow ID"].str.contains("158.65.110.24")
        Zeus_1 = data_df["Flow ID"].str.contains("192.168.3.35")
        Zeus_2 = data_df["Flow ID"].str.contains("192.168.3.25")
        Zeus_3 = data_df["Flow ID"].str.contains("192.168.3.65")
        bin_Zeus = data_df["Flow ID"].str.contains("172.29.0.116")
        Osx_trojan = data_df["Flow ID"].str.contains("172.29.0.109")
        Zero_access_1 = data_df["Flow ID"].str.contains("172.16.253.132")
        Zero_access_2 = data_df["Flow ID"].str.contains("192.168.248.165")
        Smoke_bot = data_df["Flow ID"].str.contains("10.37.130.4")
        if CTU_NERIS:

            CTU_neris = data_df["Flow ID"].str.contains("147.32.84.165")

        # SARUMAN1 = data_df['Flow ID'].str.contains('147.32.84.191')
        # SARUMAN2 = data_df['Flow ID'].str.contains('147.32.84.192')
        # SARUMAN3 = data_df['Flow ID'].str.contains('147.32.84.193')

        # SARUMAN4 = data_df['Flow ID'].str.contains('147.32.84.204')
        # SARUMAN5 = data_df['Flow ID'].str.contains('147.32.84.205')
        # SARUMAN6 = data_df['Flow ID'].str.contains('147.32.84.206')
        # SARUMAN7 = data_df['Flow ID'].str.contains('147.32.84.207')
        # SARUMAN8 = data_df['Flow ID'].str.contains('147.32.84.208')
        # SARUMAN9 = data_df['Flow ID'].str.contains('147.32.84.209')

        indx_IRC_1 = [i for i, x in enumerate(IRC_1) if x]
        indx_IRC_2 = [i for i, x in enumerate(IRC_2) if x]
        indx_IRC_3 = [i for i, x in enumerate(IRC_3) if x]
        indx_IRC_4 = [i for i, x in enumerate(IRC_4) if x]
        indx_IRC_5 = [i for i, x in enumerate(IRC_5) if x]
        indx_IRC_6 = [i for i, x in enumerate(IRC_6) if x]
        indx_IRC_7 = [i for i, x in enumerate(IRC_7) if x]
        indx_IRC_8 = [i for i, x in enumerate(IRC_8) if x]
        indx_IRC_9 = [i for i, x in enumerate(IRC_9) if x]
        indx_IRC_10 = [i for i, x in enumerate(IRC_10) if x]
        indx_IRC_11 = [i for i, x in enumerate(IRC_11) if x]
        indx_IRC_12 = [i for i, x in enumerate(IRC_12) if x]
        indx_IRC_13 = [i for i, x in enumerate(IRC_13) if x]
        indx_IRC_14 = [i for i, x in enumerate(IRC_14) if x]
        indx_IRC_15 = [i for i, x in enumerate(IRC_15) if x]

        indx_Neris = [i for i, x in enumerate(Neris) if x]
        indx_RBot = [i for i, x in enumerate(RBot) if x]
        indx_Menti = [i for i, x in enumerate(Menti) if x]
        indx_Sogou = [i for i, x in enumerate(Sogou) if x]
        indx_Murlo = [i for i, x in enumerate(Murlo) if x]
        indx_Virut = [i for i, x in enumerate(Virut) if x]
        indx_Black_hole_1 = [i for i, x in enumerate(Black_hole_1) if x]
        indx_Black_hole_2 = [i for i, x in enumerate(Black_hole_2) if x]
        indx_Black_hole_3 = [i for i, x in enumerate(Black_hole_3) if x]
        indx_TBot_1 = [i for i, x in enumerate(TBot_1) if x]
        indx_TBot_2 = [i for i, x in enumerate(TBot_2) if x]
        indx_TBot_3 = [i for i, x in enumerate(TBot_3) if x]
        indx_TBot_4 = [i for i, x in enumerate(TBot_4) if x]
        indx_Weasel_master = [i for i, x in enumerate(Weasel_master) if x]
        indx_Weasel_bot = [i for i, x in enumerate(Weasel_bot) if x]
        indx_Zeus_1 = [i for i, x in enumerate(Zeus_1) if x]
        indx_Zeus_2 = [i for i, x in enumerate(Zeus_2) if x]
        indx_Zeus_3 = [i for i, x in enumerate(Zeus_3) if x]
        indx_bin_Zeus = [i for i, x in enumerate(bin_Zeus) if x]
        indx_Osx_trojan = [i for i, x in enumerate(Osx_trojan) if x]
        indx_Zero_access_1 = [i for i, x in enumerate(Zero_access_1) if x]
        indx_Zero_access_2 = [i for i, x in enumerate(Zero_access_2) if x]
        indx_Smoke_bot = [i for i, x in enumerate(Smoke_bot) if x]
        indx_Zero_access_1 = [i for i, x in enumerate(Zero_access_1) if x]
        indx_Zero_access_2 = [i for i, x in enumerate(Zero_access_2) if x]
        indx_Smoke_bot = [i for i, x in enumerate(Smoke_bot) if x]

        if CTU_NERIS:

            indx_CTU_neris = [i for i, x in enumerate(CTU_neris) if x]

        # indx_SARUMAN1 = [i for i, x in enumerate(SARUMAN1) if x]
        # indx_SARUMAN2 = [i for i, x in enumerate(SARUMAN2) if x]
        # indx_SARUMAN3 = [i for i, x in enumerate(SARUMAN3) if x]
        # indx_SARUMAN4 = [i for i, x in enumerate(SARUMAN4) if x]
        # indx_SARUMAN5 = [i for i, x in enumerate(SARUMAN5) if x]
        # indx_SARUMAN6 = [i for i, x in enumerate(SARUMAN6) if x]
        # indx_SARUMAN7 = [i for i, x in enumerate(SARUMAN7) if x]
        # indx_SARUMAN8 = [i for i, x in enumerate(SARUMAN8) if x]
        # indx_SARUMAN9 = [i for i, x in enumerate(SARUMAN9) if x]

        total_instances = data_df.shape[0]

        if DEBUG:
            print("Total Instances:" + str(total_instances))

            print(
                "bin_IRC_1_Instances:"
                + str(len(indx_IRC_1))
                + " ---> "
                + str(round(len(indx_IRC_1) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_2_Instances:"
                + str(len(indx_IRC_2))
                + " ---> "
                + str(round(len(indx_IRC_2) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_3_Instances:"
                + str(len(indx_IRC_3))
                + " ---> "
                + str(round(len(indx_IRC_3) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_4_Instances:"
                + str(len(indx_IRC_4))
                + " ---> "
                + str(round(len(indx_IRC_4) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_5_Instances:"
                + str(len(indx_IRC_5))
                + " ---> "
                + str(round(len(indx_IRC_5) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_6_Instances:"
                + str(len(indx_IRC_6))
                + " ---> "
                + str(round(len(indx_IRC_6) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_7_Instances:"
                + str(len(indx_IRC_7))
                + " ---> "
                + str(round(len(indx_IRC_7) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_8_Instances:"
                + str(len(indx_IRC_8))
                + " ---> "
                + str(round(len(indx_IRC_8) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_9_Instances:"
                + str(len(indx_IRC_9))
                + " ---> "
                + str(round(len(indx_IRC_9) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_10_Instances:"
                + str(len(indx_IRC_10))
                + " ---> "
                + str(round(len(indx_IRC_10) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_11_Instances:"
                + str(len(indx_IRC_11))
                + " ---> "
                + str(round(len(indx_IRC_11) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_12_Instances:"
                + str(len(indx_IRC_12))
                + " ---> "
                + str(round(len(indx_IRC_12) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_13_Instances:"
                + str(len(indx_IRC_13))
                + " ---> "
                + str(round(len(indx_IRC_13) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_14_Instances:"
                + str(len(indx_IRC_14))
                + " ---> "
                + str(round(len(indx_IRC_14) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_IRC_15_Instances:"
                + str(len(indx_IRC_15))
                + " ---> "
                + str(round(len(indx_IRC_15) / total_instances * 100, 6))
                + " %"
            )

            print(
                "Neris_Instances:"
                + str(len(indx_Neris))
                + " ---> "
                + str(round(len(indx_Neris) / total_instances * 100, 6))
                + " %"
            )
            print(
                "RBot_Instances:"
                + str(len(indx_RBot))
                + " ---> "
                + str(round(len(indx_RBot) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Menti_Instances:"
                + str(len(indx_Menti))
                + " ---> "
                + str(round(len(indx_Menti) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Sogou_Instances:"
                + str(len(indx_Sogou))
                + " ---> "
                + str(round(len(indx_Sogou) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Murlo_Instances:"
                + str(len(indx_Murlo))
                + " ---> "
                + str(round(len(indx_Murlo) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Virut_Instances:"
                + str(len(indx_Virut))
                + " ---> "
                + str(round(len(indx_Virut) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Black_hole_1_Instances:"
                + str(len(indx_Black_hole_1))
                + " ---> "
                + str(round(len(indx_Black_hole_1) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Black_hole_2_Instances:"
                + str(len(indx_Black_hole_2))
                + " ---> "
                + str(round(len(indx_Black_hole_2) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Black_hole_3_Instances:"
                + str(len(indx_Black_hole_3))
                + " ---> "
                + str(round(len(indx_Black_hole_3) / total_instances * 100, 6))
                + " %"
            )
            print(
                "TBot_1_Instances:"
                + str(len(indx_TBot_1))
                + " ---> "
                + str(round(len(indx_TBot_1) / total_instances * 100, 6))
                + " %"
            )
            print(
                "TBot_2_Instances:"
                + str(len(indx_TBot_2))
                + " ---> "
                + str(round(len(indx_TBot_2) / total_instances * 100, 6))
                + " %"
            )
            print(
                "TBot_3_Instances:"
                + str(len(indx_TBot_3))
                + " ---> "
                + str(round(len(indx_TBot_3) / total_instances * 100, 6))
                + " %"
            )
            print(
                "TBot_4_Instances:"
                + str(len(indx_TBot_4))
                + " ---> "
                + str(round(len(indx_TBot_4) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Weasel_master_Instances:"
                + str(len(indx_Weasel_master))
                + " ---> "
                + str(round(len(indx_Weasel_master) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Weasel_bot_Instances:"
                + str(len(indx_Weasel_bot))
                + " ---> "
                + str(round(len(indx_Weasel_bot) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Zeus_1_Instances:"
                + str(len(indx_Zeus_1))
                + " ---> "
                + str(round(len(indx_Zeus_1) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Zeus_2_Instances:"
                + str(len(indx_Zeus_2))
                + " ---> "
                + str(round(round(len(indx_Zeus_2) / total_instances * 100, 6), 2))
                + " %"
            )
            print(
                "Zeus_3_Instances:"
                + str(len(indx_Zeus_3))
                + " ---> "
                + str(round(len(indx_Zeus_3) / total_instances * 100, 6))
                + " %"
            )
            print(
                "bin_Zeus_Instances:"
                + str(len(indx_bin_Zeus))
                + " ---> "
                + str(round(len(indx_bin_Zeus) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Osx_trojan_Instances:"
                + str(len(indx_Osx_trojan))
                + " ---> "
                + str(round(len(indx_Osx_trojan) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Zero_access_1_Instances:"
                + str(len(indx_Zero_access_1))
                + " ---> "
                + str(round(len(indx_Zero_access_1) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Zero_access_2_Instances:"
                + str(len(indx_Zero_access_2))
                + " ---> "
                + str(round(len(indx_Zero_access_2) / total_instances * 100, 6))
                + " %"
            )
            print(
                "Smoke_bot_Instances:"
                + str(len(indx_Smoke_bot))
                + " ---> "
                + str(round(len(indx_Smoke_bot) / total_instances * 100, 6))
                + " %"
            )

            if CTU_NERIS:
                print(
                    "CTU_neris_instances:"
                    + str(len(indx_CTU_neris))
                    + " ---> "
                    + str(round(len(indx_CTU_neris) / total_instances * 100, 6))
                    + " %"
                )

            # print("indx_SARUMAN1_instances:" + str(len(indx_SARUMAN1)) + " ---> "+ str(round(len(indx_SARUMAN1)/total_instances*100, 6)) + " %")
            # print("indx_SARUMAN2_instances:" + str(len(indx_SARUMAN2)) + " ---> "+ str(round(len(indx_SARUMAN2)/total_instances*100, 6)) + " %")
            # print("indx_SARUMAN3_instances:" + str(len(indx_SARUMAN3)) + " ---> "+ str(round(len(indx_SARUMAN3)/total_instances*100, 6)) + " %")
            # print("indx_SARUMAN4_instances:" + str(len(indx_SARUMAN4)) + " ---> "+ str(round(len(indx_SARUMAN4)/total_instances*100, 6)) + " %")
            # print("indx_SARUMAN5_instances:" + str(len(indx_SARUMAN5)) + " ---> "+ str(round(len(indx_SARUMAN5)/total_instances*100, 6)) + " %")
            # print("indx_SARUMAN6_instances:" + str(len(indx_SARUMAN6)) + " ---> "+ str(round(len(indx_SARUMAN6)/total_instances*100, 6)) + " %")
            # print("indx_SARUMAN7_instances:" + str(len(indx_SARUMAN7)) + " ---> "+ str(round(len(indx_SARUMAN7)/total_instances*100, 6)) + " %")
            # print("indx_SARUMAN8_instances:" + str(len(indx_SARUMAN8)) + " ---> "+ str(round(len(indx_SARUMAN8)/total_instances*100, 6)) + " %")
            # print("indx_SARUMAN9_instances:" + str(len(indx_SARUMAN9)) + " ---> "+ str(round(len(indx_SARUMAN9)/total_instances*100, 6)) + " %")

        # This cell labels the 'Label' column in the data frame to 1 where the particular botnet was found

        if CAT_FEATURES == 1:
            data.loc[:, "Label"] = "Normal"

            data.loc[indx_IRC_1, "Label"] = "IRC_1"
            data.loc[indx_IRC_2, "Label"] = "IRC_2"
            data.loc[indx_IRC_3, "Label"] = "IRC_3"
            data.loc[indx_IRC_4, "Label"] = "IRC_4"
            data.loc[indx_IRC_5, "Label"] = "IRC_5"
            data.loc[indx_IRC_6, "Label"] = "IRC_6"
            data.loc[indx_IRC_7, "Label"] = "IRC_7"
            data.loc[indx_IRC_11, "Label"] = "IRC_11"
            data.loc[indx_IRC_15, "Label"] = "IRC_15"
            data.loc[indx_Neris, "Label"] = "Neris"
            data.loc[indx_RBot, "Label"] = "RBot"
            data.loc[indx_Menti, "Label"] = "Menti"
            data.loc[indx_Sogou, "Label"] = "Sogou"
            data.loc[indx_Murlo, "Label"] = "Murlo"
            data.loc[indx_Virut, "Label"] = "Virut"
            data.loc[indx_Black_hole_1, "Label"] = "Black_hole_1"
            data.loc[indx_Black_hole_2, "Label"] = "Black_hole_2"
            data.loc[indx_Black_hole_3, "Label"] = "Black_hole_3"
            data.loc[indx_TBot_1, "Label"] = "TBot_1"
            data.loc[indx_TBot_2, "Label"] = "TBot_2"
            data.loc[indx_TBot_3, "Label"] = "TBot_3"
            data.loc[indx_TBot_4, "Label"] = "TBot_4"
            data.loc[indx_Weasel_master, "Label"] = "Weasel_master"
            data.loc[indx_Weasel_bot, "Label"] = "Weasel_bot"
            data.loc[indx_Zeus_1, "Label"] = "Zeus_1"
            data.loc[indx_Zeus_2, "Label"] = "Zeus_2"
            data.loc[indx_Zeus_3, "Label"] = "Zeus_3"
            data.loc[indx_bin_Zeus, "Label"] = "bin_Zeus"
            data.loc[indx_Osx_trojan, "Label"] = "Osx_trojan"
            data.loc[indx_Zero_access_1, "Label"] = "Zero_access_1"
            data.loc[indx_Zero_access_2, "Label"] = "Zero_access_2"
            data.loc[indx_Smoke_bot, "Label"] = "Smoke_bot"

            if CTU_NERIS:
                data.loc[indx_CTU_neris, "Label"] = "CTU_neris"

        else:

            data.loc[:, "Label"] = 1

            data.loc[indx_IRC_1, "Label"] = 1
            data.loc[indx_IRC_2, "Label"] = 2
            data.loc[indx_IRC_3, "Label"] = 3
            data.loc[indx_IRC_4, "Label"] = 4
            data.loc[indx_IRC_5, "Label"] = 5
            data.loc[indx_IRC_6, "Label"] = 6
            data.loc[indx_IRC_7, "Label"] = 7
            data.loc[indx_IRC_11, "Label"] = 8
            data.loc[indx_IRC_15, "Label"] = 9
            data.loc[indx_Neris, "Label"] = 10
            data.loc[indx_RBot, "Label"] = 11
            data.loc[indx_Menti, "Label"] = 12
            data.loc[indx_Sogou, "Label"] = 13
            data.loc[indx_Murlo, "Label"] = 14
            data.loc[indx_Virut, "Label"] = 15
            data.loc[indx_Black_hole_1, "Label"] = 16
            data.loc[indx_Black_hole_2, "Label"] = 17
            data.loc[indx_Black_hole_3, "Label"] = 18
            data.loc[indx_TBot_1, "Label"] = 19
            data.loc[indx_TBot_2, "Label"] = 20
            data.loc[indx_TBot_3, "Label"] = 21
            data.loc[indx_TBot_4, "Label"] = 22
            data.loc[indx_Weasel_master, "Label"] = 23
            data.loc[indx_Weasel_bot, "Label"] = 24
            data.loc[indx_Zeus_1, "Label"] = 25
            data.loc[indx_Zeus_2, "Label"] = 26
            data.loc[indx_Zeus_3, "Label"] = 27
            data.loc[indx_bin_Zeus, "Label"] = 28
            data.loc[indx_Osx_trojan, "Label"] = 29
            data.loc[indx_Zero_access_1, "Label"] = 30
            data.loc[indx_Zero_access_2, "Label"] = 31
            data.loc[indx_Smoke_bot, "Label"] = 32

            if CTU_NERIS:
                data.loc[indx_CTU_neris, "Label"] = 33

            if USE_ONE_BOT:

                one_bot_samples = data.loc[data["Label"] == 15].copy()

                one_bot_samples.loc[:, "Label"] = 0

                benign_samples = data.loc[data["Label"] == 1].copy()

                data = pd.concat([benign_samples, one_bot_samples]).reset_index(
                    drop=True
                )  # Augmenting with real botnets

                INPUT_FILE_NAME = INPUT_FILE_NAME + "_VIRUT"

            elif USE_FOUR_BOTS:

                # =============================================
                # We will be using 4 Bots for ISCX Dataset
                # =============================================

                # ================== BOT 1 ==================

                # bot_1_samples = data.loc[ data['Label']==15].copy()   # Virut

                # bot_1_samples.loc[:, 'Label'] = 1

                benign_samples = data.loc[data["Label"] == 0].copy()

                # data_frame = pd.concat([benign_samples, bot_1_samples]).reset_index(drop=True) #Augmenting with real botnets

                # ================ ADD BOT 1 ================

                bot_1_samples = data.loc[data["Label"] == 25].copy()  # Zeus_1

                bot_1_samples.loc[:, "Label"] = 1

                data_frame = pd.concat([benign_samples, bot_1_samples]).reset_index(
                    drop=True
                )  # Augmenting with real botnets

                # ================ ADD BOT 2 =================

                bot_2_samples = data.loc[data["Label"] == 26].copy()  # Zeus_2

                bot_2_samples.loc[:, "Label"] = 1

                data_frame = pd.concat([data_frame, bot_2_samples]).reset_index(
                    drop=True
                )  # Augmenting with real botnets

                # ================ ADD BOT 3 =================

                bot_3_samples = data.loc[data["Label"] == 27].copy()  # bin_Zeus

                bot_3_samples.loc[:, "Label"] = 1

                data_frame = pd.concat([data_frame, bot_3_samples]).reset_index(
                    drop=True
                )  # Augmenting with real botnets

                # ================ ADD BOT 4 =================

                bot_4_samples = data.loc[data["Label"] == 28].copy()  # Zeus_bin

                bot_4_samples.loc[:, "Label"] = 1

                data_frame = pd.concat([data_frame, bot_4_samples]).reset_index(
                    drop=True
                )  # Augmenting with real botnets

                data = data_frame.copy()

                INPUT_FILE_NAME = INPUT_FILE_NAME + "_4_BOTS"

            elif USE_ALL_BOTS:

                bot_samples = data.loc[data["Label"] != 0].copy()

                bot_samples.loc[:, "Label"] = 1

                benign_samples = data.loc[data["Label"] == 0].copy()

                data = pd.concat([benign_samples, bot_samples]).reset_index(
                    drop=True
                )  # Augmenting with real botnets

            # data.loc[indx_IRC_1, 'Label'] = 1
            # data.loc[indx_IRC_2, 'Label'] = 1
            # data.loc[indx_IRC_3, 'Label'] = 1
            # data.loc[indx_IRC_4, 'Label'] = 1
            # data.loc[indx_IRC_5, 'Label'] = 1
            # data.loc[indx_IRC_6, 'Label'] = 1
            # data.loc[indx_IRC_7, 'Label'] = 1
            # data.loc[indx_IRC_11, 'Label'] = 1
            # data.loc[indx_IRC_15, 'Label'] = 1
            # data.loc[indx_Neris, 'Label'] = 1
            # data.loc[indx_RBot, 'Label'] = 1
            # data.loc[indx_Menti, 'Label'] = 1
            # data.loc[indx_Sogou, 'Label'] = 1
            # data.loc[indx_Murlo, 'Label'] = 1
            # data.loc[indx_Virut, 'Label'] = 1
            # data.loc[indx_Black_hole_1, 'Label'] = 1
            # data.loc[indx_Black_hole_2, 'Label'] = 1
            # data.loc[indx_Black_hole_3, 'Label'] = 1
            # data.loc[indx_TBot_1, 'Label'] = 1
            # data.loc[indx_TBot_2, 'Label'] = 1
            # data.loc[indx_TBot_3, 'Label'] = 1
            # data.loc[indx_TBot_4, 'Label'] = 1
            # data.loc[indx_Weasel_master, 'Label'] = 1
            # data.loc[indx_Weasel_bot, 'Label'] = 1
            # data.loc[indx_Zeus_1, 'Label'] = 1
            # data.loc[indx_Zeus_2, 'Label'] = 1
            # data.loc[indx_Zeus_3, 'Label'] = 1
            # data.loc[indx_bin_Zeus, 'Label'] = 1
            # data.loc[indx_Osx_trojan, 'Label'] = 1
            # data.loc[indx_Zero_access_1, 'Label'] = 1
            # data.loc[indx_Zero_access_2, 'Label'] = 1
            # data.loc[indx_Smoke_bot, 'Label'] = 1

            # if CTU_NERIS:
            #     data.loc[indx_CTU_neris, 'Label'] = 1

            # data.loc[indx_SARUMAN1, 'Label'] = 1
            # data.loc[indx_SARUMAN2, 'Label'] = 1
            # data.loc[indx_SARUMAN3, 'Label'] = 1
            # data.loc[indx_SARUMAN4, 'Label'] = 1
            # data.loc[indx_SARUMAN5, 'Label'] = 1
            # data.loc[indx_SARUMAN6, 'Label'] = 1
            # data.loc[indx_SARUMAN7, 'Label'] = 1
            # data.loc[indx_SARUMAN8, 'Label'] = 1
            # data.loc[indx_SARUMAN9, 'Label'] = 1

    # =============================================================================================
    # Print the dataset shape before preprocessing
    # =============================================================================================

    normal = data.loc[data["Label"] == 0].copy()
    bots = data.loc[data["Label"] == 1].copy()

    print("Before Preprocesing: Total: " + str(data.shape))
    print("Before Preprocesing: Normal: " + str(normal.shape))
    print("Before Preprocesing: Bots: " + str(bots.shape))

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")

        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    # for cols in data.columns.tolist()[1:]:

    if DEBUG:

        print("Data Shape after droping NULL and INF values: ")

        print(data.shape)

    # =============================================================================================

    data_df = data.drop(["Flow ID", "Label"], axis=1)

    data_cols = data_df.columns

    # print('')

    # print(data_df.describe())

    data_df = data_df[data[data_cols] >= 0].copy()

    # print(data_df.describe())

    # print('')

    if DEBUG:

        print("data_df after removing Label column")
        print(data_df.shape)

    # =============================================================================================
    # Convert all values to float
    # =============================================================================================

    data_df = data_df.astype(float)

    if DEBUG:
        print(" Data Columns after converting to Float: " + str(data_df.columns))

    # =============================================================================================
    # Compute skew greater than 1 and less than -1 for suppressing outliers
    # =============================================================================================
    if REMOVE_SKEW:

        skew = data_df.skew(axis=0, skipna=True)

        print(skew)

        high_skew_list = skew[skew > 20].index.tolist()
        low_skew_list = skew[skew < 20].index.tolist()

        if DEBUG:

            print("low skew list" + str(low_skew_list))
            print("high skew list" + str(high_skew_list))

            # for i in high_skew_list:
            #     sns.distplot(data_df[i])

            #     print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            #     plt.show()

        data_df[high_skew_list] = np.log(data_df[high_skew_list].values + 1).copy()
        data_df[low_skew_list] = np.log(data_df[low_skew_list].values + 1).copy()

        # for i in high_skew_list:
        #     sns.distplot(data_df[i])

        #     print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        #     plt.show()

    # =============================================================================================
    # Remove any left over inf values
    # =============================================================================================

    inf_indx = data_df.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("INF values before removing: " + str(inf_indx))

    data_df = data_df.drop(inf_indx, axis=0).copy()
    data = data.drop(inf_indx, axis=0).copy()

    data_cols = data_df.columns
    data[data_cols] = data_df.copy()

    data = data.reset_index(drop=True).copy()

    if DEBUG:
        print("INF values removed and Data reindexed")

    # =============================================================================================
    # remove any columns with std = Zero
    # =============================================================================================

    if DEBUG:
        print("Data before removing std = 0 columns")
        print(data_df.shape)
        print(data.shape)

    data_df = data_df.loc[:, data_df.var() == 0]

    data = data.drop(data_df.columns, axis=1)

    if DEBUG:

        print("Data after removing std = 0 columns")
        print(data.shape)

    # =============================================================================================
    # Scale data between 0 and 1 for GAN input
    # =============================================================================================
    data_df = data.drop(["Flow ID", "Label"], axis=1)

    print(data_df.describe())

    data_df -= data_df.min()
    data_df /= data_df.max()

    print(
        "HEREEEEEEEEEEEEEEEEEEEEEEEEEEE+++++++++++++++++++++++++++++++>>>>>>>>>>>>>>>>>>>>>>"
    )

    print(data_df.describe())

    # =============================================================================================
    # Check if there is any NaN
    # =============================================================================================
    inf_indx = data_df.isnull().sum().sum()

    if DEBUG:

        print("INF values: " + str(inf_indx))

    inf_indx = data.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("Any Left over INF values: " + str(inf_indx))

    data[data_df.columns] = data_df.copy()
    # print(data.describe(include = 'all'))

    # =============================================================================================
    # Check count of bots in the dataset and save to the file
    # =============================================================================================
    # bots = data['Label'].value_counts()[1]
    # total_flows = data.shape[0]

    # print('Botnet counts are: '+ str(bots) + '(' + str(bots/total_flows * 100) + '%)\n' )

    # =============================================================================================
    data = data.drop(["Flow ID"], axis=1)

    data = round(data, 8).copy()

    if DEBUG:
        print(" Data Columns after removing Flow ID: " + str(data.columns))

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans
    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    # selected_features = ['Idle Max', 'Idle Mean', 'Packet Length Min', 'FIN Flag Count', 'FWD Init Win Bytes', 'Bwd Packet Length Min', 'Flow IAT Min', 'Idle Min', 'Subflow Fwd Bytes', 'Fwd IAT Min', 'Fwd Packet Length Min', 'Packet Length Std', 'Fwd Packets/s', 'Bwd Packets/s', 'Label']
    # data = data[selected_features].copy()

    # print(data.describe())

    # # =============================================================================================
    #     print('File: ' + str(INPUT_FILE_NAME)+ '_(Preprocessed).csv Saving ...')
    #     data.to_csv(str(PATH) + str(INPUT_FILE_NAME)+ '_(Preprocessed).csv')
    #     print('File: ' + str(INPUT_FILE_NAME)+ '_(Preprocessed).csv saved to directory')

    # # =============================================================================================
    data = data.drop(["Flow IAT Std", "Fwd IAT Std", "CWR Flag Count"], axis=1).copy()

    if USE_FEATURE_REDUCTION:
        selected_columns = top_features(data, "XGB")
        print(selected_columns)

        data_frame = data.drop(["Label"], axis=1)

        data_frame = data[selected_columns].copy()

        data_frame["Label"] = data["Label"].copy()

        data = data_frame.copy()

        print(data.shape)

    normal = data.loc[data["Label"] == 0].copy()
    bots = data.loc[data["Label"] == 1].copy()

    print("After Preprocesing: Total: " + str(data.shape))
    print("After Preprocesing: Normal: " + str(normal.shape))
    print("After Preprocesing: Bots: " + str(bots.shape))

    # =============================================================================================
    print("File: " + str(INPUT_FILE_NAME) + "_(Preprocessed).csv Saving ...")
    data.to_csv(str(PATH) + str(INPUT_FILE_NAME) + "_(Preprocessed).csv")
    print("File: " + str(INPUT_FILE_NAME) + "_(Preprocessed).csv saved to directory")

    # =============================================================================================

    return data


# =============================================================================================
# =============================================================================================
# =============================================================================================
def prepare_cic_2017_data(PATH="", INPUT_FILE_NAME="", CSV_ONE_BOT=0):

    # =============================================================================================
    # Read Data
    # =============================================================================================
    data = pd.read_csv(str(PATH) + str(INPUT_FILE_NAME), low_memory=False)

    print(
        "Processing File: " + str(INPUT_FILE_NAME) + " DATA shape: " + str(data.shape)
    )

    normal = data.loc[data["Label"] == 1].copy()
    bots = data.loc[data["Label"] == 0].copy()

    print("Before Preprocesing: Total: " + str(data.shape))
    print("Before Preprocesing: Normal: " + str(normal.shape))
    print("Before Preprocesing: Bots: " + str(bots.shape))

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")

        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    # for cols in data.columns.tolist()[1:]:

    if DEBUG:

        print("Data Shape after droping NULL and INF values: ")

        print(data.shape)

    # =============================================================================================
    # Drop all categorical features
    # =============================================================================================

    # if network_traffic:
    #     data= data.drop(['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp'], axis=1).copy()
    #     if DEBUG:
    #         print('data_df after removing categorical features')
    #         print(data.shape)

    # =============================================================================================

    data_df = data.drop(["Label"], axis=1)

    data_cols = data_df.columns

    data_df = data_df[data[data_cols] >= 0].copy()  # Remove -ve values

    # print(data_df.describe())

    if DEBUG:

        print("data_df after removing Label column")
        print(data_df.shape)

    # =============================================================================================
    # Convert all values to float
    # =============================================================================================

    data_df = data_df.astype(float)

    if DEBUG:
        print(" Data Columns after converting to Float")
        print(data_df.describe())

    # =============================================================================================
    # Compute skew greater than 1 and less than -1 for suppressing outliers
    # =============================================================================================
    if REMOVE_SKEW:

        skew = data_df.skew(axis=0, skipna=True)

        print(skew)

        high_skew_list = skew[skew > 20].index.tolist()
        low_skew_list = skew[skew < 20].index.tolist()

        if DEBUG:

            print("low skew list" + str(low_skew_list))
            print("high skew list" + str(high_skew_list))

        data_df[high_skew_list] = np.log(data_df[high_skew_list].values + 1).copy()
        data_df[low_skew_list] = np.log(data_df[low_skew_list].values + 1).copy()

    # =============================================================================================
    # Remove any left over inf values
    # =============================================================================================

    inf_indx = data_df.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("INF values before removing: " + str(inf_indx))

    data_df = data_df.drop(inf_indx, axis=0).copy()
    data = data.drop(inf_indx, axis=0).copy()

    data_cols = data_df.columns
    data[data_cols] = data_df.copy()

    data = data.reset_index(drop=True).copy()

    if DEBUG:
        print("INF values removed and Data reindexed")

    # =============================================================================================
    # remove any columns with std = Zero
    # =============================================================================================

    if DEBUG:
        print("Data before removing std = 0 columns")
        print(data_df.shape)
        print(data.shape)

    data_df = data_df.loc[:, data_df.var() == 0]

    data = data.drop(data_df.columns, axis=1)

    if DEBUG:

        print("Data after removing std = 0 columns")
        print(data.shape)

    # =============================================================================================
    # Scale data between 0 and 1 for GAN input
    # =============================================================================================

    data_df = data.drop(["Label"], axis=1)

    # print(data_df.describe())

    data_df -= data_df.min()
    data_df /= data_df.max()

    # print(data_df.describe())

    # =============================================================================================
    # Check if there is any NaN
    # =============================================================================================
    inf_indx = data_df.isnull().sum().sum()

    if DEBUG:

        print("INF values: " + str(inf_indx))

    inf_indx = data.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("Any Left over INF values: " + str(inf_indx))

    data[data_df.columns] = data_df.copy()
    # print(data.describe(include = 'all'))

    # =============================================================================================
    # Check count of bots in the dataset and save to the file
    # =============================================================================================
    # bots = data['Label'].value_counts()[1]
    # total_flows = data.shape[0]

    # print('Botnet counts are: '+ str(bots) + '(' + str(bots/total_flows * 100) + '%)\n' )

    if (
        network_traffic
    ):  # This is just for testing purposes. I can test this code with another tabular dataset so this section will be skipped in that case by disabling the network_traffic flag in the start of this code.
        # =============================================================================================
        # Choose only Flow ID and Label columns for labeling purposes in another data frame
        # =============================================================================================

        data_df = data["Label"].copy()

        # =============================================================================================
        # remove any columns with all values = Zero
        # =============================================================================================

        data = data.loc[:, (data != 0).any(axis=0)]

        # IRC = data_df['Flow ID'].str.contains('192.168.2.112-131.202.243.84|192.168.5.122-198.164.30.2|192.168.2.110-192.168.5.122|192.168.4.118-192.168.5.122|192.168.2.113-192.168.5.122|192.168.1.103-192.168.5.122|192.168.4.120-192.168.5.122|192.168.2.112-192.168.2.110|192.168.2.112-192.168.4.120|192.168.2.112-192.168.1.103|192.168.2.112-192.168.2.113|192.168.2.112-192.168.4.118|192.168.2.112-192.168.2.109|192.168.2.112-192.168.2.105|192.168.1.105-192.168.5.122')
        # =============================================================================================

        if DEBUG:
            print(" Data Columns after removing Flow ID: " + str(data.columns))

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans
    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    # selected_features = ['Idle Max', 'Idle Mean', 'Packet Length Min', 'FIN Flag Count', 'FWD Init Win Bytes', 'Bwd Packet Length Min', 'Flow IAT Min', 'Idle Min', 'Subflow Fwd Bytes', 'Fwd IAT Min', 'Fwd Packet Length Min', 'Packet Length Std', 'Fwd Packets/s', 'Bwd Packets/s', 'Label']
    # data = data[selected_features].copy()

    # print(data.describe())
    data = round(data, 8)

    if USE_FEATURE_REDUCTION:

        selected_columns = top_features(data, "XGB")
        print(selected_columns)

        data_frame = data.drop(["Label"], axis=1)

        data_frame = data[selected_columns].copy()

        data_frame["Label"] = data["Label"].copy()

        data = data_frame.copy()

        print(data.shape)

    normal = data.loc[data["Label"] == 1].copy()
    bots = data.loc[data["Label"] == 0].copy()

    print("After Preprocesing: Total: " + str(data.shape))
    print("After Preprocesing: Normal: " + str(normal.shape))
    print("After Preprocesing: Bots: " + str(bots.shape))

    # data= data.drop(['TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FlowIATMin', 'FwdIATMin', 'BwdIATMin', 'FwdPSHFlags', 'BwdHeaderLength', 'FINFlagCount', 'SYNFlagCount', 'RSTFlagCount', 'ECEFlagCount', 'FwdHeaderLength', 'SubflowFwdPackets', 'SubflowBwdPackets' , 'SubflowBwdBytes', 'act_data_pkt_fwd', 'ActiveMean', 'ActiveStd', 'ActiveMax', 'ActiveMin', 'IdleStd'], axis=1).copy()
    # =============================================================================================
    print("File: " + str(INPUT_FILE_NAME) + "_(Preprocessed).csv Saving ...")
    data.to_csv(str(PATH) + str(INPUT_FILE_NAME) + "_(Preprocessed).csv")
    print("File: " + str(INPUT_FILE_NAME) + "_(Preprocessed).csv saved to directory")

    # =============================================================================================

    # data= data.drop(['Total Fwd Packets'], axis=1).copy()
    # print(data.columns)

    return data


# =============================================================================================
# =============================================================================================
# =============================================================================================


def prepare_cic_2018_data(PATH="", INPUT_FILE_NAME="", CSV_ONE_BOT=0):

    # =============================================================================================
    # Read Data
    # =============================================================================================

    data = pd.read_csv(str(PATH) + str(INPUT_FILE_NAME), low_memory=False)

    print(
        "Processing File: " + str(INPUT_FILE_NAME) + " DATA shape: " + str(data.shape)
    )

    print("The shape of dataset is: ", data.shape)

    data["Label"] = data["Label"].replace(["Benign", "Bot"], [1, 0])

    normal = data.loc[data["Label"] == 1].copy()
    bots = data.loc[data["Label"] == 0].copy()

    print("Before Preprocesing: Total: " + str(data.shape))
    print("Before Preprocesing: Normal: " + str(normal.shape))
    print("Before Preprocesing: Bots: " + str(bots.shape))

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # print(data.describe())

    # replace inf with nan and then drop the rows with nans

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")

        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    # for cols in data.columns.tolist()[1:]:

    if DEBUG:

        print("Data Shape after droping NULL and INF values: ")

        print(data.shape)

    # =============================================================================================
    # Drop all categorical features
    # =============================================================================================

    data = data.drop(["Dst Port", "Timestamp", "Protocol"], axis=1).copy()
    if DEBUG:
        print("data_df after removing categorical features")
        print(data.shape)

    # =============================================================================================

    data_df = data.drop(["Label"], axis=1)

    data_cols = data_df.columns

    data_df = data_df[data[data_cols] >= 0].copy()  # Remove -ve values

    # print(data_df.describe())

    if DEBUG:

        print("data_df after removing Label column")
        print(data_df.shape)

    # =============================================================================================
    # Convert all values to float
    # =============================================================================================

    data_df = data_df.astype(float)

    if DEBUG:
        print(" Data Columns after converting to Float")
        print(data_df.describe())

    # =============================================================================================
    # Compute skew greater than 1 and less than -1 for suppressing outliers
    # =============================================================================================
    if REMOVE_SKEW:

        skew = data_df.skew(axis=0, skipna=True)

        print(skew)

        high_skew_list = skew[skew > 20].index.tolist()
        low_skew_list = skew[skew < 20].index.tolist()

        if DEBUG:

            print("low skew list" + str(low_skew_list))
            print("high skew list" + str(high_skew_list))

        data_df[high_skew_list] = np.log(data_df[high_skew_list].values + 1).copy()
        data_df[low_skew_list] = np.log(data_df[low_skew_list].values + 1).copy()

    # =============================================================================================
    # Remove any left over inf values
    # =============================================================================================

    inf_indx = data_df.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("INF values before removing: " + str(inf_indx))

    data_df = data_df.drop(inf_indx, axis=0).copy()
    data = data.drop(inf_indx, axis=0).copy()

    data_cols = data_df.columns
    data[data_cols] = data_df.copy()

    data = data.reset_index(drop=True).copy()

    if DEBUG:
        print("INF values removed and Data reindexed")

    # =============================================================================================
    # remove any columns with std = Zero
    # =============================================================================================

    if DEBUG:
        print("Data before removing std = 0 columns")
        print(data_df.shape)
        print(data.shape)

    data_df = data_df.loc[:, data_df.var() == 0]

    data = data.drop(data_df.columns, axis=1)

    if DEBUG:

        print("Data after removing std = 0 columns")
        print(data.shape)

    # =============================================================================================
    # Scale data between 0 and 1 for GAN input
    # =============================================================================================

    data_df = data.drop(["Label"], axis=1)

    # print(data_df.describe())

    print(data_df.describe())

    print(
        "HEREEEEEEEEEEEEEEEEEEEEEEEEEEE+++++++++++++++++++++++++++++++>>>>>>>>>>>>>>>>>>>>>>"
    )

    data_df -= data_df.min()
    data_df /= data_df.max()

    print(data_df.describe())

    # =============================================================================================
    # Check if there is any NaN
    # =============================================================================================
    inf_indx = data_df.isnull().sum().sum()

    if DEBUG:

        print("INF values: " + str(inf_indx))

    inf_indx = data.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("Any Left over INF values: " + str(inf_indx))

    data[data_df.columns] = data_df.copy()

    data_df = data["Label"].copy()

    # =============================================================================================
    # remove any columns with all values = Zero
    # =============================================================================================

    data = data.loc[:, (data != 0).any(axis=0)]

    # =============================================================================================

    if DEBUG:
        print(" Data Columns after removing Flow ID: " + str(data.columns))

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans
    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    # selected_features = ['Idle Max', 'Idle Mean', 'Packet Length Min', 'FIN Flag Count', 'FWD Init Win Bytes', 'Bwd Packet Length Min', 'Flow IAT Min', 'Idle Min', 'Subflow Fwd Bytes', 'Fwd IAT Min', 'Fwd Packet Length Min', 'Packet Length Std', 'Fwd Packets/s', 'Bwd Packets/s', 'Label']
    # data = data[selected_features].copy()

    # print(data.describe())
    data = round(data, 8)

    if USE_FEATURE_REDUCTION:
        selected_columns = top_features(data, "XGB")
        print(selected_columns)

        data_frame = data.drop(["Label"], axis=1)

        data_frame = data[selected_columns].copy()

        data_frame["Label"] = data["Label"].copy()

        data = data_frame.copy()

        print(data.shape)

    # ========== Extract a Chunk of Bots ============================================================

    normal = data.loc[data["Label"] == 1].copy()
    bots = data.loc[data["Label"] == 0].copy()

    print("After Preprocesing: Bots: " + str(bots.shape))

    bots = bots[0 : 512 * 5]

    # =============================================================================================

    data = pd.concat([bots, normal]).reset_index(drop=True).copy()

    normal = data.loc[data["Label"] == 1].copy()
    bots = data.loc[data["Label"] == 0].copy()

    print("After Preprocesing: Total: " + str(data.shape))
    print("After Preprocesing: Normal: " + str(normal.shape))
    print("After Preprocesing: Bots Chunk: " + str(bots.shape))

    # data= data.drop(['TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FlowIATMin', 'FwdIATMin', 'BwdIATMin', 'FwdPSHFlags', 'BwdHeaderLength', 'FINFlagCount', 'SYNFlagCount', 'RSTFlagCount', 'ECEFlagCount', 'FwdHeaderLength', 'SubflowFwdPackets', 'SubflowBwdPackets' , 'SubflowBwdBytes', 'act_data_pkt_fwd', 'ActiveMean', 'ActiveStd', 'ActiveMax', 'ActiveMin', 'IdleStd'], axis=1).copy()
    # =============================================================================================
    print("File: " + str(INPUT_FILE_NAME) + "_(Preprocessed).csv Saving ...")
    data.to_csv(str(PATH) + str(INPUT_FILE_NAME) + "_(Preprocessed).csv")
    print("File: " + str(INPUT_FILE_NAME) + "_(Preprocessed).csv saved to directory")

    # =============================================================================================

    return data


def prepare_UNSW_IoT(PATH="", INPUT_FILE_NAME=""):

    # =============================================================================================
    # Read Data
    # =============================================================================================
    data = pd.read_csv(str(PATH) + str(INPUT_FILE_NAME), low_memory=False)

    print(
        "Processing File: " + str(INPUT_FILE_NAME) + " DATA shape: " + str(data.shape)
    )

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")

        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    # for cols in data.columns.tolist()[1:]:

    if DEBUG:

        print("Data Shape after droping NULL and INF values: ")

        print(data.shape)

    # =============================================================================================
    # Drop all categorical features
    # =============================================================================================

    data = data.drop(
        [
            "pkSeqID",
            "saddr",
            "sport",
            "daddr",
            "dport",
            "seq",
            "category",
            "subcategory",
            "method",
        ],
        axis=1,
    ).copy()
    if DEBUG:
        print("data_df after removing categorical features")
        print(data.shape)

    # =============================================================================================

    data_df = data.drop(["Label"], axis=1)

    data_cols = data_df.columns

    print(data_df.describe)

    data_df = data_df[data_df[data_cols] >= 0].copy()

    if DEBUG:

        print("data_df after removing Label column")
        print(data_df.shape)

    # =============================================================================================
    # Convert all values to float
    # =============================================================================================

    data_df = data_df.astype(float)

    if DEBUG:
        print(" Data Columns after converting to Float: " + str(data_df.columns))

    # =============================================================================================
    # Compute skew greater than 1 and less than -1 for suppressing outliers
    # =============================================================================================
    if REMOVE_SKEW:

        skew = data_df.skew(axis=0, skipna=True)

        print(skew)

        high_skew_list = skew[skew > 10].index.tolist()
        low_skew_list = skew[skew < 10].index.tolist()

        if DEBUG:

            print("low skew list" + str(low_skew_list))
            print("high skew list" + str(high_skew_list))

        data_df[high_skew_list] = np.log(data_df[high_skew_list].values + 1).copy()
        data_df[low_skew_list] = np.log(data_df[low_skew_list].values + 1).copy()

    # =============================================================================================
    # Remove any left over inf values
    # =============================================================================================

    inf_indx = data_df.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("INF values before removing: " + str(inf_indx))

    data_df = data_df.drop(inf_indx, axis=0).copy()
    data = data.drop(inf_indx, axis=0).copy()

    data_cols = data_df.columns
    data[data_cols] = data_df.copy()

    data = data.reset_index(drop=True).copy()

    if DEBUG:
        print("INF values removed and Data reindexed")

    # =============================================================================================
    # remove any columns with std = Zero
    # =============================================================================================

    if DEBUG:
        print("Data before removing std = 0 columns")
        print(data_df.shape)
        print(data.shape)

    data_df = data_df.loc[:, data_df.var() == 0]

    data = data.drop(data_df.columns, axis=1)

    if DEBUG:

        print("Data after removing std = 0 columns")
        print(data.shape)

    # =============================================================================================
    # Scale data between -1 and 1 for GAN input
    # =============================================================================================
    data_df = data.drop(["Label"], axis=1)

    data_df -= data_df.min()
    data_df /= data_df.max()

    # =============================================================================================
    # Check if there is any NaN
    # =============================================================================================
    inf_indx = data_df.isnull().sum().sum()

    if DEBUG:

        print("INF values: " + str(inf_indx))

    inf_indx = data.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("Any Left over INF values: " + str(inf_indx))

    data[data_df.columns] = data_df.copy()
    data = round(data, 8).copy()

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans
    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )
    data = data.rename(columns={"Label": "Label"})

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    print(data.describe())

    # =============================================================================================
    print("File: " + str(INPUT_FILE_NAME) + "_(Preprocessed).csv Saving ...")
    data.to_csv(str(PATH) + str(INPUT_FILE_NAME) + "_(Preprocessed).csv")
    print("File: " + str(INPUT_FILE_NAME) + "_(Preprocessed).csv saved to directory")

    # =============================================================================================
    data = data.drop(["response_body_len", "is_sm_ips_ports"], axis=1).copy()

    return data


# =============================================================================================
# =============================================================================================
# =============================================================================================


def prepare_DARKNET_2020_data(PATH="", INPUT_FILE_NAME="", CSV_ONE_BOT=0):

    # =============================================================================================
    # Read Data
    # =============================================================================================

    data = pd.read_csv(str(PATH) + str(INPUT_FILE_NAME), low_memory=False)

    print(
        "Processing File: " + str(INPUT_FILE_NAME) + " DATA shape: " + str(data.shape)
    )

    print("The shape of dataset is: ", data.shape)

    data["Label"] = data["Label"].replace(["Benign", "Darknet"], [0, 1])

    normal = data.loc[data["Label"] == 0].copy()
    bots = data.loc[data["Label"] == 1].copy()

    print("Before Preprocesing: Total: " + str(data.shape))
    print("Before Preprocesing: Normal: " + str(normal.shape))
    print("Before Preprocesing: Darknet: " + str(bots.shape))

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # print(data.describe())

    # replace inf with nan and then drop the rows with nans

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")

        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    # for cols in data.columns.tolist()[1:]:

    if DEBUG:

        print("Data Shape after droping NULL and INF values: ")

        print(data.shape)

    # print(data.head())

    # =============================================================================================
    # Drop all categorical features
    # =============================================================================================

    data = data.drop(
        [
            "Flow ID",
            "Src IP",
            "Src Port",
            "Dst IP",
            "Dst Port",
            "Protocol",
            "Timestamp",
            "Label_1",
        ],
        axis=1,
    ).copy()

    if DEBUG:
        print("data_df after removing categorical features")
        print(data.shape)

    # =============================================================================================

    data_df = data.drop(["Label"], axis=1)

    data_cols = data_df.columns

    data_df = data_df[data[data_cols] >= 0].copy()  # Remove -ve values

    # print(data_df.describe())

    if DEBUG:

        print("data_df after removing Label column")
        print(data_df.shape)

    # =============================================================================================
    # Convert all values to float
    # =============================================================================================

    data_df = data_df.astype(float)

    if DEBUG:
        print(" Data Columns after converting to Float")
        print(data_df.describe())

    # =============================================================================================
    # Compute skew greater than 1 and less than -1 for suppressing outliers
    # =============================================================================================
    if REMOVE_SKEW:

        skew = data_df.skew(axis=0, skipna=True)

        print(skew)

        high_skew_list = skew[skew > 20].index.tolist()
        low_skew_list = skew[skew < 20].index.tolist()

        if DEBUG:

            print("low skew list" + str(low_skew_list))
            print("high skew list" + str(high_skew_list))

        data_df[high_skew_list] = np.log(data_df[high_skew_list].values + 1).copy()
        data_df[low_skew_list] = np.log(data_df[low_skew_list].values + 1).copy()

    # =============================================================================================
    # Remove any left over inf values
    # =============================================================================================

    inf_indx = data_df.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("INF values before removing: " + str(inf_indx))

    data_df = data_df.drop(inf_indx, axis=0).copy()
    data = data.drop(inf_indx, axis=0).copy()

    data_cols = data_df.columns
    data[data_cols] = data_df.copy()

    data = data.reset_index(drop=True).copy()

    if DEBUG:
        print("INF values removed and Data reindexed")

    # =============================================================================================
    # remove any columns with std = Zero
    # =============================================================================================

    if DEBUG:
        print("Data before removing std = 0 columns")
        print(data_df.shape)
        print(data.shape)

    data_df = data_df.loc[:, data_df.var() == 0]

    data = data.drop(data_df.columns, axis=1)

    if DEBUG:

        print("Data after removing std = 0 columns")
        print(data.shape)

    # =============================================================================================
    # Scale data between 0 and 1 for GAN input
    # =============================================================================================

    data_df = data.drop(["Label"], axis=1)

    # print(data_df.describe())

    data_df -= data_df.min()
    data_df /= data_df.max()

    # print(data_df.describe())

    # =============================================================================================
    # Check if there is any NaN
    # =============================================================================================
    inf_indx = data_df.isnull().sum().sum()

    if DEBUG:

        print("INF values: " + str(inf_indx))

    inf_indx = data.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("Any Left over INF values: " + str(inf_indx))

    data[data_df.columns] = data_df.copy()

    data_df = data["Label"].copy()

    # =============================================================================================
    # remove any columns with all values = Zero
    # =============================================================================================

    data = data.loc[:, (data != 0).any(axis=0)]

    # =============================================================================================

    if DEBUG:
        print(" Data Columns after removing Flow ID: " + str(data.columns))

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans
    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    if DEBUG:
        print("Data Shape after droping NULL and INF values: ")
        print(data.shape)

    # selected_features = ['Idle Max', 'Idle Mean', 'Packet Length Min', 'FIN Flag Count', 'FWD Init Win Bytes', 'Bwd Packet Length Min', 'Flow IAT Min', 'Idle Min', 'Subflow Fwd Bytes', 'Fwd IAT Min', 'Fwd Packet Length Min', 'Packet Length Std', 'Fwd Packets/s', 'Bwd Packets/s', 'Label']
    # data = data[selected_features].copy()

    # print(data.describe())
    data = round(data, 8)

    if USE_FEATURE_REDUCTION:
        selected_columns = top_features(data, "XGB")
        print(selected_columns)

        data_frame = data.drop(["Label"], axis=1)

        data_frame = data[selected_columns].copy()

        data_frame["Label"] = data["Label"].copy()

        data = data_frame.copy()

        print(data.shape)

    # ========== Extract a Chunk of Bots ============================================================

    NonTor = data.loc[data["Label"] == "Non-Tor"].copy()
    NonVPN = data.loc[data["Label"] == "NonVPN"].copy()
    VPN = data.loc[data["Label"] == "VPN"].copy()
    Tor = data.loc[data["Label"] == "Tor"].copy()

    print("After Preprocesing: NonTor: " + str(NonTor.shape))
    print("After Preprocesing: NonVPN: " + str(NonVPN.shape))
    print("After Preprocesing: VPN: " + str(VPN.shape))
    print("After Preprocesing: Tor: " + str(Tor.shape))

    bots = pd.concat([Tor, VPN]).reset_index(drop=True)
    normal = pd.concat([NonTor, NonVPN]).reset_index(drop=True)

    bots["Label"] = 1
    normal["Label"] = 0

    # bots = bots[0:512*5]

    # =============================================================================================

    data = pd.concat([bots, normal]).reset_index(drop=True).copy()

    # normal = data.loc[ data['Label']==0 ].copy()
    # bots = data.loc[ data['Label']==1 ].copy()

    # print('After Preprocesing: Total: ' + str(data.shape))
    # print('After Preprocesing: Normal: ' + str(normal.shape))
    # print('After Preprocesing: Darknet Chunk: ' + str(bots.shape))

    # data= data.drop(['TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FlowIATMin', 'FwdIATMin', 'BwdIATMin', 'FwdPSHFlags', 'BwdHeaderLength', 'FINFlagCount', 'SYNFlagCount', 'RSTFlagCount', 'ECEFlagCount', 'FwdHeaderLength', 'SubflowFwdPackets', 'SubflowBwdPackets' , 'SubflowBwdBytes', 'act_data_pkt_fwd', 'ActiveMean', 'ActiveStd', 'ActiveMax', 'ActiveMin', 'IdleStd'], axis=1).copy()
    # =============================================================================================
    print("File: " + str(INPUT_FILE_NAME) + "_(Preprocessed).csv Saving ...")
    data.to_csv(str(PATH) + str(INPUT_FILE_NAME) + "_(Preprocessed).csv")
    print("File: " + str(INPUT_FILE_NAME) + "_(Preprocessed).csv saved to directory")

    # =============================================================================================

    return data


# =============================================================================================
# =============================================================================================
# =============================================================================================

from sklearn.metrics import recall_score, precision_score, roc_auc_score
from header import *
import header
import importlib
importlib.reload(header)  # For reloading after making changes


DEBUG = 1


# ==================================================================

def plot_kde(real_samples, gen_samples, normal_traffic, with_class=False):

    plt.style.use('seaborn-white')

    data_cols = real_samples.columns

    len_of_columns = len(data_cols)

    if with_class:
        len_of_columns = len_of_columns - 1

    cols = 6
    rows = len_of_columns // cols + 1

    print(rows)

    i = 0
    # print(real_samples['response_body_len'].describe())

    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(28, 38))

    for row in range(rows):
        for col in range(cols):

            sns.kdeplot(data=real_samples, x=data_cols[i], color='#e70a16', ax=axs[row]
                        [col], label='Real Bots', bw_adjust=2, fill=True, linewidth=1.4)
            axs[0][0].legend()
            # axs[0][0].axes.yaxis.set_visible(False)

            sns.kdeplot(data=gen_samples, x=data_cols[i], color='#0074f3', ax=axs[row]
                        [col], label='GAN Bots', bw_adjust=2, fill=True, linewidth=1.4)
            axs[0][0].legend()

            sns.kdeplot(data=normal_traffic, x=data_cols[i], color='#017102', ax=axs[row]
                        [col], label='Normal Traffic', bw_adjust=2, fill=True, linewidth=1.4)
            axs[0][0].legend()
            # axs[0][0].axes.yaxis.set_visible(False)

            i = i + 1

            if i >= len_of_columns:
                break
        if i >= len_of_columns:
            break
    i = 0

    # plt.legend(['Real Bots', 'GAN, Bots'])
    # #axs[0][0].legend()

    # if save_fig:

    #     plt.savefig(figs_path + TODAY + '/' + cache_prefix + '_KDE' + str(list_log_iteration[-1]) + '.pdf', dpi=600)

    # fig.suptitle('RMS = ' +str(rms) + '\nRMS(min) = ' + str(min(rms_list)), fontsize=16)

    plt.show()

    plt.close(fig)

# ==================================================================


def recall(preds, dtrain):
    labels = dtrain.get_label()
    return 'recall',  recall_score(labels, np.round(preds))

# ==================================================================


def precision(preds, dtrain):
    return 'precision',  precision_score(labels, np.round(preds))

# ==================================================================


def roc_auc(preds, dtrain):
    return 'roc_auc',  roc_auc_score(labels, preds)

# ==================================================================


def perf_measure(y_pred, y_test):

    TP = np.sum((y_pred == 1) & (y_test == 1))
    TN = np.sum((y_pred == 0) & (y_test == 0))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))
    # evasions = np.where((y_pred == 0) & (y_test == 1), 1, 0)
    # print([i for i, x in enumerate(evasions) if x])
    return TP, TN, FP, FN

# ==================================================================


def SimpleMetrics(y_pred, y_test):
    TP, TN, FP, FN = perf_measure(y_pred, y_test)
    ACC = (TP + TN) / (TP + TN + FP + FN)

    # Reporting
    print('Confusion Matrix')
    display(pd.DataFrame([[TN, FP], [FN, TP]], columns=[
            'Pred 0', 'Pred 1'], index=['True 0', 'True 1']))

    return ACC
# ==================================================================


def SimpleAccuracy(y_pred, y_test):

    ACC = SimpleMetrics(y_pred, y_test)

    print('Accuracy: ' + str(ACC))

    return ACC
# ==================================================================


def SimpleRecall(y_pred, y_test):
    TP, TN, FP, FN = perf_measure(y_pred, y_test)
    RCL = TP / (TP + FN)

    print('Recall: ' + str(RCL))

    # print( 'Recall: {}'.format( round(RCL,4) ))
    return RCL

# ==================================================================


def get_data_batch(train, batch_size, seed):
    # # random sampling - some samples will have excessively low or high sampling, but easy to implement
    # np.random.seed(seed)
    # x = train.loc[ np.random.choice(train.index, batch_size) ].values
    # print("seed is ======>>>> " + str(seed))
    # iterate through shuffled indices, so every sample gets covered evenly
    start_i = (batch_size * seed) % len(train)
    # print("start_i is ======>>>> " + str(start_i))
    stop_i = start_i + batch_size
    # print("stop_i is ======>>>> " + str(stop_i))
    shuffle_seed = (batch_size * seed) // len(train)
    # print("shuffle_seed is ======>>>> " + str(shuffle_seed))
    np.random.seed(shuffle_seed)
    # wasteful to shuffle every time
    train_ix = np.random.choice(
        list(train.index), replace=False, size=len(train))
    # duplicate to cover ranges past the end of the set
    train_ix = list(train_ix) + list(train_ix)
    x = train.loc[train_ix[start_i: stop_i]].values

    x = pd.DataFrame(x)
    x.columns = train.columns
    return_matrix = np.reshape(x, (batch_size, -1))
    # print(return_matrix)
    return return_matrix

# ==================================================================


def c2st(X, y, clf=LogisticRegression(), loss=hamming_loss, bootstraps=10):
    """
    Perform Classifier Two Sample Test (C2ST) [1].

    This test estimates if a target is predictable from features by comparing the loss of a classifier learning
    the true target with the distribution of losses of classifiers learning a random target with the same average.

    The null hypothesis is that the target is independent of the features - therefore the loss of a classifier learning
    to predict the target should not be different from the one of a classifier learning independent, random noise.

    Input:
        - `X` : (n,m) matrix of features
        - `y` : (n,) vector of target - for now only supports binary target
        - `clf` : instance of sklearn compatible classifier (default: `LogisticRegression`)
        - `loss` : sklearn compatible loss function (default: `hamming_loss`)
        - `bootstraps` : number of resamples for generating the loss scores under the null hypothesis

    Return: (
        loss value of classifier predicting `y`,
        loss values of bootstraped random targets,
        p-value of the test
    )

    Usage:
    >>> emp_loss, random_losses, pvalue = c2st(X, y)

    Plotting H0 and target loss:
    >>>bins, _, __ = plt.hist(random_losses)
    >>>med = np.median(random_losses)
    >>>plt.plot((med,med),(0, max(bins)), 'b')
    >>>plt.plot((emp_loss,emp_loss),(0, max(bins)), 'r--')

    [1] Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv preprint arXiv:1610.06545.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    emp_loss = loss(y_test, y_pred)

    bs_losses = []
    y_bar = np.mean(y)

    for b in range(bootstraps+1):
        y_random = np.random.binomial(1, y_bar, size=y.shape[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y_random)
        y_pred_bs = clf.fit(X_train, y_train).predict(X_test)
        bs_losses += [loss(y_test, y_pred_bs)]
    pc = stats.percentileofscore(sorted(bs_losses), emp_loss) / 100.
    pvalue = pc if pc < y_bar else 1 - pc
    return emp_loss, np.array(bs_losses), pvalue
# ==================================================================


def Evaluate_Parameter_old(x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2, classifier='xgb', EVALUATION_PARAMETER=''):

    REAL_CONCAT_GEN_SET = np.vstack([x, g_z])

    REAL_CONCAT_GEN_SET_LABELS = np.hstack(
        [np.zeros(int(len(x))), np.ones(int(len(g_z)))])

    # Use half of each real and generated set for training
    dtrain = np.vstack([x[:int(len(x) / 2)], g_z[:int(len(g_z) / 2)]])
    # synthetic labels
    dlabels = np.hstack(
        [np.zeros(int(len(x) / 2)), np.ones(int(len(g_z) / 2))])
    # Use the other half of each set for testing
    dtest = np.vstack([x[int(len(x) / 2):], g_z[int(len(g_z) / 2):]])
    y_test = dlabels  # Labels for test samples will be the same as the labels for training samples, assuming even batch sizes

    # print(dtrain.shape)
    # print(dtest.shape)

    if(classifier == 'XGB'):
        print('Evaluation ---->> XBG')
        clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False)

        # print(c2st(REAL_CONCAT_GEN_SET, REAL_CONCAT_GEN_SET_LABELS, clf=clf))
        clf.fit(dtrain, y_test)
        y_pred = clf.predict(dtest)

    if ALL_CLASSIFIERS:
        if (classifier == 'DT'):
            print('Evaluation ---->> DT')
            clf = DecisionTreeClassifier()
            clf = clf.fit(dtrain, y_test)
            y_pred = clf.predict(dtest)
        elif (classifier == 'RF'):
            print('Evaluation ---->> RF')
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(dtrain, y_test)
            y_pred = clf.predict(dtest)
        elif (classifier == 'LR'):
            print('Evaluation ---->> LR')
            logreg = LogisticRegression(max_iter=10000000)
            logreg.fit(dtrain, y_test)
            y_pred = logreg.predict(dtest)
        elif (classifier == 'KNN'):
            print('Evaluation ---->> KNN')
            knn_classifier = KNeighborsClassifier(n_neighbors=5)
            knn_classifier.fit(dtrain, y_test)
            y_pred = knn_classifier.predict(dtest)
        elif (classifier == 'NB'):
            print('Evaluation ---->> NB')
            gnb = GaussianNB()
            # Train the model using the training sets
            gnb.fit(dtrain, y_test)
            y_pred = gnb.predict(dtest)

        # elif (classifier=='SVM'):
        #     if DEBUG:
        #         print('Evaluation ---->> SVM >>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        #     svclassifier = SVC(kernel='linear')
        #     svclassifier.fit(dtrain, y_test)
        #     y_pred = svclassifier.predict(dtest)

    y_pred = np.round(y_pred)
    # print(y_pred)
    # return '{:.2f}'.format(SimpleAccuracy(y_pred, y_test)) # assumes
    # balanced real and generated datasets
    # assumes balanced real and generated datasets
    if EVALUATION_PARAMETER == 'Recall':
        # SimpleMetrics(y_pred, y_test)
        # return SimpleAccuracy(y_pred, y_test)
        return SimpleRecall(y_pred, y_test)
    return [SimpleAccuracy(y_pred, y_test), SimpleRecall(y_pred, y_test)]

# ==================================================================


def Evaluate_Parameter(x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2, classifier='xgb', EVALUATION_PARAMETER=''):

    rcl = 0
    acc = 0

    g_z = pd.DataFrame(g_z)

    g_z.columns = x.columns

    REAL_CONCAT_GEN_SET = np.vstack([x, g_z])
    REAL_CONCAT_GEN_SET = pd.DataFrame(REAL_CONCAT_GEN_SET)
    REAL_CONCAT_GEN_SET.columns = x.columns

    REAL_CONCAT_GEN_SET_LABELS = np.hstack(
        [np.zeros(int(len(x))), np.ones(int(len(g_z)))])

    REAL_CONCAT_GEN_SET['Label'] = REAL_CONCAT_GEN_SET_LABELS

    # REAL_CONCAT_GEN_SET = REAL_CONCAT_GEN_SET.sample(frac=1).reset_index(drop=True)

    REAL_CONCAT_GEN_SET_LABELS = REAL_CONCAT_GEN_SET['Label'].values

    # print(pd.DataFrame(REAL_CONCAT_GEN_SET_LABELS))

    # REAL_CONCAT_GEN_SET.to_csv(str(DATA_SET_PATH) + 'GAN' + 'GAN_REAL_CONCAT.csv')
    # print('File: ' + 'GAN' + '_AUG_DATA_SET.csv saved to directory')
# =====================================================================

    # print(REAL_CONCAT_GEN_SET.describe())

    if (classifier == 'XGB'):
        # clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False)

    if ALL_CLASSIFIERS:
        if (classifier == 'DT'):
            clf = DecisionTreeClassifier()

        elif (classifier == 'NB'):
            clf = GaussianNB()

        elif (classifier == 'RF'):
            # clf = RandomForestClassifier(n_estimators=100)
            clf = RandomForestClassifier()

        elif (classifier == 'LR'):
            # clf = LogisticRegression(max_iter=10000)
            clf = LogisticRegression(max_iter=10000000)

        elif (classifier == 'KNN'):
            # clf = KNeighborsClassifier(n_neighbors=5)
            clf = KNeighborsClassifier()

    for i in range(10):  # 10-folds
        X_train, X_test, y_train, y_test = train_test_split(
            REAL_CONCAT_GEN_SET, REAL_CONCAT_GEN_SET_LABELS, test_size=0.3, random_state=i)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_ = accuracy_score(y_test, y_pred)
        rcl_ = recall_score(y_test, y_pred)

        # print(i+1, '[Acc:', acc_,',Rcl:' ,rcl_, ']')

        acc += acc_
        rcl += rcl_

    acc = acc/(i+1)
    rcl = rcl/(i+1)

    print(classifier, ': [Acc:', acc, ', Rcl:', rcl, ']')

    return [acc, rcl]
# ==================================================================


def ConfusionMatrix(y_pred, y_test):
    TP, TN, FP, FN = perf_measure(y_pred, y_test)
    from IPython.display import display
    print('Confusion Matrix')
    display(pd.DataFrame([[TN, FP], [FN, TP]], columns=[
            'Pred Normal', 'Pred Bot'], index=['Normal', 'Bot']))
# ==================================================================


def clsfr_train_test(X_train, y_train, X_test, y_test, accu_list=[], rcl_list=[], prec_list=[], f1_list=[], clf=0):
    clf.fit(X_train, y_train)
    y_pred = np.round(clf.predict(X_test))

    accu = accuracy_score(y_test, y_pred)
    rcl = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # TP, TN, FP, FN = perf_measure(y_pred, y_test)

    # accu = ( TP + TN ) / ( TP + TN + FP + FN )
    # rcl  = TP  / ( TP + FN )
    # print(TP, TP + FP)
    # prec = TP / ( TP + FP )
    # f1 = 2 * ( prec * rcl) / ( prec + rcl)

    accu = round(accu * 100, 2)
    rcl = round(rcl * 100, 2)
    prec = round(prec * 100, 2)
    f1 = round(f1 * 100, 2)

    accu_list.append(accu)
    rcl_list.append(rcl)
    prec_list.append(prec)
    f1_list.append(f1)

    ConfusionMatrix(y_pred, y_test)

    print('Accuracy: ' + str(accu_list) + str('%'))
    print('Recall: ' + str(rcl_list) + str('%'))
    print('Precision: ' + str(prec_list) + str('%'))
    print('F1: ' + str(f1_list) + str('%') + '\n\n')

# ===============================================================================================================================
# ===============================================================================================================================
# ===============================================================================================================================


def run_all_classifiers(X_train, y_train, X_test, y_test, accu_list=[], rcl_list=[], prec_list=[], f1_list=[]):

    print('Running XGB ...')
    clsfr_train_test(X_train, y_train, X_test, y_test, accu_list, rcl_list, prec_list,
                     f1_list, clf=XGBClassifier(eval_metric='logloss', use_label_encoder=False))

    print('Running DT ...')
    clsfr_train_test(X_train, y_train, X_test, y_test, accu_list,
                     rcl_list, prec_list, f1_list, clf=DecisionTreeClassifier())

    print('Running NB ...')
    clsfr_train_test(X_train, y_train, X_test, y_test, accu_list,
                     rcl_list, prec_list, f1_list, clf=GaussianNB())

    print('Running RF ...')
    clsfr_train_test(X_train, y_train, X_test, y_test, accu_list, rcl_list,
                     prec_list, f1_list, clf=RandomForestClassifier(n_estimators=100))

    print('Running LR ...')
    clsfr_train_test(X_train, y_train, X_test, y_test, accu_list, rcl_list,
                     prec_list, f1_list, clf=LogisticRegression(max_iter=10000000))

    print('Running KNN ...')
    clsfr_train_test(X_train, y_train, X_test, y_test, accu_list, rcl_list,
                     prec_list, f1_list, clf=KNeighborsClassifier(n_neighbors=5))


# ===============================================================================================================================
# ===============================================================================================================================
# ===============================================================================================================================
def generate_gan_data(x, labels=[], weight_or_epoch_number=0, data_dim=0, FULL_CACHE_PATH='',  GAN_type='', TODAY='', DATA_SIZE=0):

    with_class = False
    NOISE_SIZE = 100

    print(GAN_type)

    if GAN_type == 'GAN':
        base_n_count = 256
        gen_model, disc_model, comb_model = define_models_GAN(
            100, data_dim, base_n_count)

    elif GAN_type == 'keras_GAN':
        gen_model = GAN(IMG_SHAPE=data_dim).generator

    elif GAN_type == 'CGAN':
        with_class = True
        base_n_count = 64
        gen_model, disc_model, comb_model = define_models_CGAN(
            100, data_dim, 1, base_n_count)

    elif GAN_type == 'WGAN':

        gen_model = WGAN(IMG_SHAPE=data_dim).generator

        # base_n_count = 128
        # gen_model, disc_model, comb_model = define_models_WGAN(100, data_dim, base_n_count)

    elif GAN_type == 'WCGAN':
        with_class = True
        base_n_count = 128

        gen_model, disc_model, comb_model = define_models_CGAN(
            NOISE_SIZE, data_dim, 1, base_n_count, type='Wasserstein')

    print('Generating ' + GAN_type + '-bots')

    gen_model.load_weights(FULL_CACHE_PATH + TODAY + '/' + GAN_type +
                           '_generator_model_weights_step_' + str(weight_or_epoch_number)+'.h5')

    np.random.seed(20)

    z = np.random.normal(size=(DATA_SIZE, 100))

    if USE_UNIFORM_NOISE:

        z = np.random.uniform(size=(DATA_SIZE, NOISE_SIZE))

    if with_class:

        g_z = gen_model.predict([z, labels])
    else:
        g_z = gen_model.predict(z)

    # g_z -= g_z.min()
    # g_z /= g_z.max()

    df = pd.DataFrame(g_z).copy()

    if GAN_type == 'GAN' or GAN_type == 'WGAN':

        df.columns = x.columns[:-1]

    elif GAN_type == 'CGAN' or GAN_type == 'WCGAN':

        df.columns = x.columns

    df['Label'] = 1  # Label = 1 (For Black box Attack)

    return df
# ===============================================================================================================================
# ===============================================================================================================================
# ===============================================================================================================================


def augment_bots(X_train, y_train, bots, cols, GAN_type='', DATA_SET_PATH='', classifier=''):
    df = pd.DataFrame(X_train)
    df.columns = cols[:-1]

    df['Label'] = y_train
    if DEBUG:

        BOT_COUNTS = df['Label'].value_counts()[1]
        BENIGN_COUNTS = df['Label'].value_counts()[0]

        print('Bots in dataset:')
        print(BOT_COUNTS)

        print('Normal in dataset:')
        print(BENIGN_COUNTS)

        print('Dataset before aug:')
        print(df.shape)

    # for i in range(10):

    df = pd.concat([df, bots]).reset_index(
        drop=True)  # Augmenting with real botnets

    # df.loc[df[df.columns] >0.5 ] = 1  # For Husnain Data

    gen_data_set = df
# ===============================================================================================================================
    gen_data_set.to_csv(str(DATA_SET_PATH) + classifier +
                        '_' + GAN_type + '_AUG_DATA_SET.csv')
    print('File: ' + GAN_type + '_AUG_DATA_SET.csv saved to directory')
# ===============================================================================================================================

    X_train = gen_data_set[cols[:-1]].values
    y_train = gen_data_set['Label'].values

    if DEBUG:

        BOT_COUNTS = gen_data_set['Label'].value_counts()[1]
        BENIGN_COUNTS = gen_data_set['Label'].value_counts()[0]

        print('Bots in dataset:')
        print(BOT_COUNTS)

        print('Normal in dataset:')
        print(BENIGN_COUNTS)

        print('Dataset after aug:')
        print(gen_data_set.shape)

    return X_train, y_train
# ===============================================================================================================================
# ===============================================================================================================================
# ===============================================================================================================================


def augment_bots_in_test_set(X_test, y_test, bots, cols):
    df = pd.DataFrame(bots)
    # df.columns = cols[:-1]

    # df['Label'] = y_test
    # if DEBUG:

    #     BOT_COUNTS = df['Label'].value_counts()[1]

    #     print('Bots in dataset:')
    #     print(BOT_COUNTS)

    #     print('Dataset before aug:')
    #     print(df.shape)

    # for i in range(10):

    # df = pd.concat([df, bots]).reset_index(drop=True) #Augmenting with real botnets

    gen_data_set = df

    X_test = gen_data_set[cols[:-1]].values
    y_test = gen_data_set['Label'].values

    # if DEBUG:

    #     BOT_COUNTS = gen_data_set['Label'].value_counts()[1]

    #     print('Bots in dataset:')
    #     print(BOT_COUNTS)

    #     print('Dataset after aug:')
    #     print(gen_data_set.shape)

    return X_test, y_test
# ===============================================================================================================================
# ===============================================================================================================================
# ===============================================================================================================================
# def collect_evasions():

#     SimpleMetrics(y_pred, y_test)
#     evasions = np.where((y_pred == 0) & (y_test == 1), 1, 0)
#     # print([i for i, x in enumerate(evasions) if x])

#     ev_list = [i for i, x in enumerate(evasions) if x]

#     evasions_list.extend(ev_list)
#     # evasions_list = list(dict.fromkeys(evasions_list))

#     print('Indices of Elements to be added: ' + str(ev_list))

#     # print('evasion_list--> unrepeated: ' + str(evasions_list))

#     print('evasion_list size --> : ' + str(len(evasions_list)) + '\n')

#     df = dfEvasions

#     for i in evasions_list:

#         # print('\n' + str(test_set[i]) + '\n')
#         # print('Length of this sample is: ' + str(len(test_set[i]))+ '\n')

#         df = df.append(dict(zip(df.columns, test_set[i])), ignore_index=True)
#         # dfEvasions = dfEvasions.append(dict(zip(dfEvasions.columns, test_set[i])), ignore_index=True)

#     # print('Df: \n' + str(df) + '\n\n')
#     # print('Evasions df: \n' + str(dfEvasions) + '\n\n')

#     dfEvasions = pd.concat([dfEvasions, df])

#     # dfEvasions = inverse_transform(dfEvasions)

#     # print('Evasions df After Concat: \n' + str(dfEvasions) + '\n\n')


#     # print(dfEvasions.describe(include = 'all'))
#     print('=======================================>>>>>>>>>>>>>>>>>>>>>>>>')

#         dfEvasions.to_csv(DATA_SET_PATH + str(classifier) +'_evasions.csv')


def predict_clf(G_Z, test_Normal, test_Bots, clf, ONLY_GZ=False):

    pred_G_Z_clf = clf.predict(G_Z)
    Ev_GZ_Bot_clf = round(
        sum(pred_G_Z_clf) / G_Z.shape[0], 4
    )

    if ONLY_GZ == False:
        pred_Normal_clf = clf.predict(test_Normal)
        pred_Bots_clf = clf.predict(test_Bots)

        N_acc_clf = round(
            sum(pred_Normal_clf) / test_Normal.shape[0], 4
        )

        Ev_Real_Bot_clf = round(
            sum(pred_Bots_clf) / test_Bots.shape[0], 4
        )  # predict bot being bot. If it maintains near 0 then it means it is bot because bot has been labeled as 1.

    if ONLY_GZ == True:
        return [Ev_GZ_Bot_clf]

    return [N_acc_clf, Ev_Real_Bot_clf]

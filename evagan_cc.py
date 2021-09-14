from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from scipy.sparse import data
from header import *
import header
import importlib
importlib.reload(header)  # For reloading after making changes


class EVAGAN_CC():
    def __init__(self, IMG_SHAPE=0):
        # Input shape
        self.img_shape = IMG_SHAPE
        self.num_classes = 1
        self.latent_dim = 100
        optimizer = Adam(0.0002, 0.5)
        losses = [
            'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'
        ]
        comb_losses = ['binary_crossentropy', 'binary_crossentropy']
        # tf.compat.v1.disable_eager_execution()
        # Build the generator
        self.generator = self.build_generator()
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses, optimizer=optimizer)
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim, ))
        label = Input(shape=(1, ))
        img = self.generator([noise, label])
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, bot, normal = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, bot])
        self.combined.compile(loss=comb_losses, optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.img_shape))
        model.add(Activation('relu'))
        print(model.metrics_names)
        noise = Input(shape=(self.latent_dim, ))
        label = Input(shape=(1, ), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes,
                                              self.latent_dim)(label))
        model_input = tf.keras.layers.multiply([noise, label_embedding])

        # model_input = multiply([noise, label_embedding])
        img = model(model_input)

        model.summary()

        return Model([noise, label], img)

    def build_discriminator(self):
        model = Sequential()
        # model.add(Dense(256, input_dim=self.img_shape))
        # model.add(LeakyReLU(alpha=0.2))
        # # model.add(Dropout(0.25))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128, input_dim=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64, input_dim=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        img = Input(shape=self.img_shape)
        # Extract feature representation
        features = model(img)
        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        bot = Dense(1, activation="sigmoid")(features)
        normal = Dense(1, activation="sigmoid")(features)

        # Dense(self.num_classes, activation="softmax")(features)
        model.summary()
        return Model(img, [validity, bot, normal])

    def train(self, model_components):
        [
            cache_prefix, with_class, starting_step, train, Train, Test,
            data_cols, data_dim, label_cols, label_dim, rand_noise_dim,
            nb_steps, batch_size, k_d, k_g, critic_pre_train_steps,
            log_interval, learning_rate, base_n_count, CACHE_PATH, FIGS_PATH,
            show, comb_loss, disc_loss_generated, disc_loss_real, xgb_losses,
            dt_losses, nb_losses, knn_losses, rf_losses, lr_losses, test_size,
            epoch_list_disc_loss_real, epoch_list_disc_loss_generated,
            epoch_list_comb_loss, gpu_device, EVALUATION_PARAMETER, TODAY
        ] = model_components
        batch_iteration = 0
        total_time_so_far = 0
        list_batch_iteration = []
        list_log_iteration = []
        list_disc_loss_real = []
        list_disc_loss_generated = []
        list_comb_loss = []
        list_disc_perf = []
        Disc_G_Z_Real_acc_ = []
        Disc_G_Z_N_acc_ = []
        Disc_G_Z_B_acc_ = []
        Disc_N_acc_ = []
        Disc_B_acc_ = []
        xgb_acc = []
        xgb_rcl = []
        xgb_c2st = []
        epoch_list_disc_loss_real = []
        epoch_list_disc_loss_generated = []
        epoch_list_gen_loss = []
        epoch_list_disc_acc_real = []
        epoch_list_disc_acc_generated = []
        epoch_list_gen_acc = []
        dt_acc = []
        dt_rcl = []
        dt_c2st = []
        nb_acc = []
        nb_rcl = []
        nb_c2st = []
        rf_acc = []
        rf_rcl = []
        rf_c2st = []
        lr_acc = []
        lr_rcl = []
        lr_c2st = []
        knn_acc = []
        knn_rcl = []
        knn_c2st = []
        svm_acc = []
        svm_rcl = []
        svm_c2st = []

        pred_G_Z_XGB = []
        pred_G_Z_DT = []
        pred_G_Z_NB = []
        pred_G_Z_RF = []
        pred_G_Z_LR = []
        pred_G_Z_KNN = []

        pred_total_XGB = []
        pred_total_DT = []
        pred_total_NB = []
        pred_total_RF = []
        pred_total_LR = []
        pred_total_KNN = []

        GEN_Validity = []
        NORMAL_Est = []
        REAL_BOT_Eva = []
        FAKE_BOT_Eva = []

        list_loss_real_bot = []
        list_loss_fake_bot = []
        list_loss_real_normal = []
        list_loss_g = []

        x = 0
        g_z = 0
        acc_ = 0
        rcl_ = 0
        best_xgb_acc_index = 0
        best_xgb_rcl_index = 0
        best_dt_acc_index = 0
        best_dt_rcl_index = 0
        best_nb_acc_index = 0
        best_nb_rcl_index = 0
        best_rf_acc_index = 0
        best_rf_rcl_index = 0
        best_lr_acc_index = 0
        best_lr_rcl_index = 0
        best_knn_acc_index = 0
        best_knn_rcl_index = 0
        # Directory
        os.mkdir(CACHE_PATH + TODAY)
        print("Directory '% s' created" % TODAY)
        os.mkdir(FIGS_PATH + TODAY)
        print("Directory '% s' created" % TODAY)
        epoch_number = 0
        log_iteration = 0
        if DEBUG:
            print(cache_prefix, batch_size, base_n_count)
        i = 0
        # ---------------------------------------------------------------------------------------------------------------
        mean = 0
        stdv = 1
        # ---------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------
        if DEBUG:
            print("======================================================")
            print("Batch Size Selected -------->>>>>>> " + str(batch_size))
            print("======================================================")
        Bots = Train.loc[Train['Label'] == 0].copy()
        Normal = Train.loc[Train['Label'] == 1].copy()
        print(Normal.shape)
        print(Bots.shape)

        t_b = Bots[0:int(Bots.shape[0] * .7)]
        T_n = Normal[0:int(Normal.shape[0] * .7)]
        print('T_n: ' + str(T_n.shape))
        T = pd.concat([t_b, T_n
                       ]).reset_index(drop=True)  # Augmenting with real botnets
        shuffled_T_B = T.sample(frac=1).reset_index(drop=True)
        shuffled_Bots = shuffled_T_B.loc[shuffled_T_B['Label'] == 0].copy()
        shuffled_Normal = shuffled_T_B.loc[shuffled_T_B['Label'] == 1].copy(
        )
        print(shuffled_Bots.shape, shuffled_Normal.shape)
        # shuffled_T_B.to_csv('shuffled_T_B.csv')
        # print('File: ' + 'shuffled_T_B.csv saved to directory')

        # print(shuffled_Normal.columns)

        test_Normal = Normal[int(
            Normal.shape[0] * .7):Normal.shape[0]]
        test_Bots = Bots[int(
            Bots.shape[0] * .7):Bots.shape[0]]

        if ESTIMATE_CLASSIFIERS:
            print("Estimating Classifiers..")
            XGB = XGBClassifier(
                eval_metric='logloss', use_label_encoder=False)
            DT = DecisionTreeClassifier()
            NB = GaussianNB()
            RF = RandomForestClassifier()
            LR = LogisticRegression(max_iter=10000000)
            KNN = KNeighborsClassifier()

            XGB.fit(shuffled_T_B[data_cols], shuffled_T_B['Label'])
            DT.fit(shuffled_T_B[data_cols], shuffled_T_B['Label'])
            NB.fit(shuffled_T_B[data_cols], shuffled_T_B['Label'])
            RF.fit(shuffled_T_B[data_cols], shuffled_T_B['Label'])
            LR.fit(shuffled_T_B[data_cols], shuffled_T_B['Label'])
            KNN.fit(shuffled_T_B[data_cols], shuffled_T_B['Label'])

            G_Z = test_Normal.copy()

            pred_total_XGB = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                         test_Bots[data_cols], XGB, ONLY_GZ=False)
            pred_total_DT = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                        test_Bots[data_cols], DT, ONLY_GZ=False)
            pred_total_NB = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                        test_Bots[data_cols], NB, ONLY_GZ=False)
            pred_total_RF = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                        test_Bots[data_cols], RF, ONLY_GZ=False)
            pred_total_LR = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                        test_Bots[data_cols], LR, ONLY_GZ=False)
            pred_total_KNN = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                         test_Bots[data_cols], KNN, ONLY_GZ=False)

        print("Starting GAN Training..")
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        start_log_time = time.time()

        for i in range(starting_step, starting_step + nb_steps):
            # real = np.random.uniform(low=0.999, high=1.0, size=batch_size)
            # fake = np.random.uniform(low=0, high=0.001, size=batch_size)
            # print('Training Disciminator----------------->>>\n')
            for j in range(k_d):
                np.random.seed(i + j)
                z = np.random.normal(mean,
                                     stdv,
                                     size=(batch_size, rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(mean,
                                          stdv,
                                          size=(batch_size, rand_noise_dim))
                t_b = get_data_batch(shuffled_Bots[0:int(shuffled_Bots.shape[0] * .7)],
                                     batch_size,
                                     seed=i + j)
                T_b = get_data_batch(shuffled_Normal[0:int(shuffled_Normal.shape[0] * .7)],
                                     batch_size,
                                     seed=i + j)
                labels = t_b['Label']

                # print("<><><><><><><><><><><><><><><><><><><>")
                # print(labels.value_counts())
                # print("<><><><><><><><><><><><><><><><><><><>")

                Labels = T_b['Label']
                # LABELS = T_B['Label']
                # sampled_labels = np.random.randint(0, 1, (batch_size, 1))
                g_z = self.generator.predict([z, labels])
                # d_l_g = self.discriminator.train_on_batch(g_z,            [fake, Labels, labels])
                # d_l_N = self.discriminator.train_on_batch(T_b[data_cols], [real, Labels, labels])
                # d_l_B = self.discriminator.train_on_batch(t_b[data_cols], [real, Labels, labels])
                d_l_g = self.discriminator.train_on_batch(
                    g_z, [fake, labels, Labels])
                d_l_N = self.discriminator.train_on_batch(
                    T_b[data_cols], [real, labels, Labels])
                d_l_B = self.discriminator.train_on_batch(
                    t_b[data_cols], [real, labels, Labels])
                # d_loss = (d_l_g[0] + d_l_N[0] +  d_l_B[0]) /3
                # d_l_g = self.discriminator.train_on_batch(g_z, [fake, sampled_labels])
                # d_l_r = self.discriminator.train_on_batch(T_b[data_cols], [real, Labels])
                # print('d_l_g: ' + str(d_l_g[0]) + ', d_l_r: ' + str(d_l_r[0]))
                # print('D(G_z)_acc: ' + str(d_l_g[3] * 100) + '%' + ', D(R)_acc: ' + str(d_l_r[3] * 100) + '%' + ' , Acc: ' + str(d_l_r[4] * 100) + '%')
                # d_loss = 0.5 * (d_l_g[0] + d_l_r[0])
            for j in range(k_g):
                np.random.seed(i + j)
                z = np.random.normal(mean,
                                     stdv,
                                     size=(batch_size, rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(mean,
                                          stdv,
                                          size=(batch_size, rand_noise_dim))
                # g_loss = self.combined.train_on_batch([z, labels], [real,  Labels, labels])
                g_loss = self.combined.train_on_batch([z, labels],
                                                      [real, labels])
                # print(g_loss)
            # loss = round(loss, 6)
            # comb_loss.append(loss)
            # print('D_loss: ' + str(d_l_g[0] +  d_l_r[0]) + ', G_Loss: ' + str(g_loss[0]) )
            # print('D(G_z)_acc: ' + str(d_l_g[3] * 100) + '%' + ', D(R)_acc: ' + str(d_l_r[3] * 100) + '%' + ' , Acc: ' + str(d_l_r[4] * 100) + '%')
            list_batch_iteration.append(batch_iteration)
            # print('d_l_g: ' + str(d_l_g[0]) + '  d_l_N: ' + str(d_l_N[0]) + ' d_l_B: ' + str(d_l_B[0]) + '  g_loss: ' + str(g_loss[0]))
            # print('d_l_g: ' + str(d_l_g[0]) + '  d_l_N: ' + str(d_l_N[0]) + '  d_l_B: ' + str(d_l_B[0]) + '  g_loss: ' + str(g_loss[0]))
            # print ("%d [D loss: %f, Real_Fake_acc.: %.2f%%, Normal_Bot_acc: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            # print ("%d [D loss: %f, D_acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[1], 100*d_loss[4], g_loss[0]))
            # ---------------------------------------------------------------------------------------------------------------
            # Determine xgb loss each step, after training generator and discriminator
            if not i % log_interval:  # 2x faster than testing each step...
                list_log_iteration.append(log_iteration)
                log_iteration = log_iteration + 1

                epoch_list_disc_loss_real.append(
                    d_l_B[0])  # epoch_list_disc_loss_real
                epoch_list_disc_loss_generated.append(
                    d_l_g[0])  # epoch_list_disc_loss_gen
                epoch_list_gen_loss.append(
                    g_loss[0])  # epoch_list_gen_loss

                list_loss_real_bot.append(d_l_B[0])
                list_loss_fake_bot.append(d_l_g[0])
                list_loss_real_normal.append(d_l_N[0])
                list_loss_g.append(g_loss[0])

                x = get_data_batch(train, test_size, seed=i)
                z = np.random.normal(size=(test_size, rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(size=(test_size, rand_noise_dim))
                labels = x['Label']
                g_z = self.generator.predict([z, labels])
                # g_z -= g_z.min()
                # g_z /= g_z.max()
                x = x[data_cols]
                # pred = self.discriminator.predict(Test[data_cols]).copy()
                # ====================Testing on 30% of train data ================================

                z = np.random.normal(
                    size=(test_Bots.shape[0], rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(size=(test_Bots.shape[0],
                                                rand_noise_dim))
                G_z = self.generator.predict([z, test_Bots['Label']])
                G_Z = pd.DataFrame(G_z).copy()
                print('G_Z.shape: ' + str(G_Z.shape))
                G_Z.columns = data_cols
                G_Z['Label'] = 0
                pred_G_Z = self.discriminator.predict(G_Z[data_cols])
                pred_Normal = self.discriminator.predict(
                    test_Normal[data_cols])
                pred_Bots = self.discriminator.predict(
                    test_Bots[data_cols])
                G_Z_Real_acc_ = round(sum(pred_G_Z[0]) / G_Z.shape[0], 4)
                Ev_GZ_Bot = round(
                    sum(pred_G_Z[1]) / G_Z.shape[0], 4
                )  # predict generated bot being real. If it maintains near 1 then it means it is bot.
                N_acc_ = round(
                    sum(pred_Normal[2]) / test_Normal.shape[0], 4
                )  # predict normal being normal. If it maintains near 1 then it means it is normal becasue normal has been labeled as 0.
                Ev_Real_Bot = round(
                    sum(pred_Bots[1]) / test_Bots.shape[0], 4
                )  # predict bot being bot. If it maintains near 0 then it means it is bot because bot has been labeled as 1.
                Disc_G_Z_Real_acc_ = np.append(Disc_G_Z_Real_acc_,
                                               G_Z_Real_acc_)
                Disc_G_Z_B_acc_ = np.append(Disc_G_Z_B_acc_, Ev_GZ_Bot)
                Disc_N_acc_ = np.append(Disc_N_acc_, N_acc_)
                Disc_B_acc_ = np.append(Disc_B_acc_, Ev_Real_Bot)

                # print("+++++++++++++++++++++++++")
                # print("GEN_Real/Fake" + str(Disc_G_Z_Real_acc_))
                # print("Gen_Eva" + str(Disc_G_Z_B_acc_))
                # print("Normal_Est" + str(Disc_N_acc_))
                # print("Real_Eva" + str(Disc_B_acc_))
                # print("+++++++++++++++++++++++++")

# ----------------------------------------------------------------------------------
                if SHOW_TIME:
                    end_log_time = time.time()
                    log_interval_time = end_log_time - start_log_time
                    start_log_time = time.time()
                    total_time_so_far += log_interval_time
                    if DEBUG:
                        print("log_iteration: " + str(log_iteration) + "/" +
                              str(nb_steps // log_interval))
                    # print("Time taken so far: " + str(total_time_so_far)  + " seconds")
                    total_time = total_time_so_far / log_iteration * nb_steps // log_interval
                    if DEBUG:
                        print("Average time per log_iteration: " +
                              str(total_time_so_far / log_iteration))
                    time_left = round((total_time - total_time_so_far) / 3600,
                                      2)
                    time_unit = 'hours'
                    if time_left < 1:
                        time_left = round(time_left * 60, 2)
                        time_unit = 'minutes'
                    print("Time left = " + str(time_left) + " " + time_unit)
                    print('Total Time Taken: ' +
                          str(round(total_time_so_far / 60, 1)) + ' minutes')

                # save model checkpoints
                # model_checkpoint_base_name = CACHE_PATH + TODAY + \
                #     '/' + cache_prefix + '_{}_model_weights_step_{}.h5'
                # self.generator.save_weights(
                #     model_checkpoint_base_name.format('self.generator',
                #                                       epoch_number))
                # self.discriminator.save_weights(
                #     model_checkpoint_base_name.format('discriminator',
                #                                       epoch_number))
                print('Epoch#: ' + str(epoch_number) + ' completed')
                epoch_number = epoch_number + 1

        GEN_VALIDITY = pd.DataFrame(
            Disc_G_Z_Real_acc_, columns=['GEN_VALIDITY'])
        FAKE_BOT_EVA = pd.DataFrame(
            Disc_G_Z_B_acc_, columns=['FAKE_BOT_EVA'])
        REAL_NORMAL_EST = pd.DataFrame(
            Disc_N_acc_, columns=['REAL_NORMAL_EST'])
        REAL_BOT_EVA = pd.DataFrame(Disc_B_acc_, columns=['REAL_BOT_EVA'])

        D_Loss_Real_Bot = pd.DataFrame(
            list_loss_real_bot, columns=['D_Loss_Real_Bot'])
        D_Loss_Fake_Bot = pd.DataFrame(
            list_loss_fake_bot, columns=['D_Loss_Fake_Bot'])
        D_Loss_Real_Normal = pd.DataFrame(
            list_loss_real_normal, columns=['D_Loss_Real_Normal'])
        G_Loss = pd.DataFrame(list_loss_g, columns=['G_Loss'])

        time_taken = pd.DataFrame(
            [round(total_time_so_far / 60, 1)], columns=['Time'])

        if ESTIMATE_CLASSIFIERS:

            pred_G_Z_XGB = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                       test_Bots[data_cols], XGB, ONLY_GZ=True)
            pred_G_Z_DT = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                      test_Bots[data_cols], DT, ONLY_GZ=True)
            pred_G_Z_NB = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                      test_Bots[data_cols], NB, ONLY_GZ=True)
            pred_G_Z_RF = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                      test_Bots[data_cols], RF, ONLY_GZ=True)
            pred_G_Z_LR = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                      test_Bots[data_cols], LR, ONLY_GZ=True)
            pred_G_Z_KNN = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                       test_Bots[data_cols], KNN, ONLY_GZ=True)

            pred_total_XGB = pd.DataFrame(
                pred_total_XGB, columns=['XGB'])
            pred_total_DT = pd.DataFrame(
                pred_total_DT, columns=['DT'])
            pred_total_NB = pd.DataFrame(
                pred_total_NB, columns=['NB'])
            pred_total_RF = pd.DataFrame(
                pred_total_RF, columns=['RF'])
            pred_total_LR = pd.DataFrame(
                pred_total_LR, columns=['LR'])
            pred_total_KNN = pd.DataFrame(
                pred_total_KNN, columns=['KNN'])

            pred_G_Z_XGB = pd.DataFrame(
                pred_G_Z_XGB, columns=['pred_G_Z_XGB'])
            pred_G_Z_DT = pd.DataFrame(
                pred_G_Z_DT, columns=['pred_G_Z_DT'])
            pred_G_Z_NB = pd.DataFrame(
                pred_G_Z_NB, columns=['pred_G_Z_NB'])
            pred_G_Z_RF = pd.DataFrame(
                pred_G_Z_RF, columns=['pred_G_Z_RF'])
            pred_G_Z_LR = pd.DataFrame(
                pred_G_Z_LR, columns=['pred_G_Z_LR'])
            pred_G_Z_KNN = pd.DataFrame(
                pred_G_Z_KNN, columns=['pred_G_Z_KNN'])

            frames = [GEN_VALIDITY,
                      FAKE_BOT_EVA, REAL_NORMAL_EST, REAL_BOT_EVA, D_Loss_Real_Bot, D_Loss_Fake_Bot, D_Loss_Real_Normal, G_Loss, pred_total_XGB, pred_total_DT, pred_total_NB, pred_total_RF, pred_total_LR, pred_total_KNN, pred_G_Z_XGB, pred_G_Z_DT, pred_G_Z_NB, pred_G_Z_RF, pred_G_Z_LR, pred_G_Z_KNN, time_taken]
        else:
            frames = [GEN_VALIDITY, FAKE_BOT_EVA, REAL_NORMAL_EST, REAL_BOT_EVA,
                      D_Loss_Real_Bot, D_Loss_Fake_Bot, D_Loss_Real_Normal, G_Loss, time_taken]

        LISTS = pd.concat(frames, sort=True, axis=1).to_csv(
            CACHE_PATH + TODAY + '/EVAGAN_CC_LISTS.csv')

        epoch_number = 0
        log_iteration = 0
        epoch_number = 0

        return [
            best_xgb_acc_index, best_xgb_rcl_index, best_dt_acc_index,
            best_dt_rcl_index, best_nb_acc_index, best_nb_rcl_index,
            best_rf_acc_index, best_rf_rcl_index, best_lr_acc_index,
            best_lr_rcl_index, best_knn_acc_index, best_knn_rcl_index
        ]


def train_EVAGAN_CC(arguments,
                    train,
                    Train,
                    Test,
                    data_cols,
                    label_cols=[],
                    seed=0,
                    starting_step=0):
    [
        rand_noise_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps,
        log_interval, learning_rate, base_n_count, CACHE_PATH, FIGS_PATH, show,
        test_size, gpu_device, EVALUATION_PARAMETER, TODAY
    ] = arguments
    with_class = False
    # np.random.seed(seed)     # set random seed
    data_dim = len(data_cols)
    # print('data_dim: ', data_dim)
    # print('data_cols: ', data_cols)
    label_dim = 0
    cache_prefix = 'ACGAN_CC'
    comb_loss, disc_loss_generated, disc_loss_real, xgb_losses, dt_losses, nb_losses,  knn_losses, rf_losses, lr_losses, epoch_list_disc_loss_real, epoch_list_disc_loss_generated, epoch_list_comb_loss = [
    ], [], [], [], [], [], [], [], [], [], [], []
    model_components = [
        cache_prefix, with_class, starting_step, train, Train, Test, data_cols,
        data_dim, label_cols, label_dim, rand_noise_dim, nb_steps, batch_size,
        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate,
        base_n_count, CACHE_PATH, FIGS_PATH, show, comb_loss,
        disc_loss_generated, disc_loss_real, xgb_losses, dt_losses, nb_losses,
        knn_losses, rf_losses, lr_losses, test_size, epoch_list_disc_loss_real,
        epoch_list_disc_loss_generated, epoch_list_comb_loss, gpu_device,
        EVALUATION_PARAMETER, TODAY
    ]
    [
        best_xgb_acc_index, best_xgb_rcl_index, best_dt_acc_index,
        best_dt_rcl_index, best_nb_acc_index, best_nb_rcl_index,
        best_rf_acc_index, best_rf_rcl_index, best_lr_acc_index,
        best_lr_rcl_index, best_knn_acc_index, best_knn_rcl_index
    ] = EVAGAN_CC(IMG_SHAPE=len(data_cols)).train(model_components)
    return [
        best_xgb_acc_index, best_xgb_rcl_index, best_dt_acc_index,
        best_dt_rcl_index, best_nb_acc_index, best_nb_rcl_index,
        best_rf_acc_index, best_rf_rcl_index, best_lr_acc_index,
        best_lr_rcl_index, best_knn_acc_index, best_knn_rcl_index
    ]

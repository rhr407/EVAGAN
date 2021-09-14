from header import *
import header
import importlib
importlib.reload(header)  # For reloading after making changes


class ACGAN_CC():
    def __init__(self, IMG_SHAPE=0):
        # Input shape
        self.img_shape = IMG_SHAPE
        self.num_classes = 2
        self.latent_dim = 100
        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'binary_crossentropy']
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
        valid, normal_or_bot = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, normal_or_bot])
        self.combined.compile(loss=losses, optimizer=optimizer)

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
        model.summary()
        print(model.metrics_names)
        noise = Input(shape=(self.latent_dim, ))
        label = Input(shape=(1, ), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes,
                                              self.latent_dim)(label))
        model_input = tf.keras.layers.Multiply()([noise, label_embedding])
        # model_input = multiply([noise, label_embedding])
        img = model(model_input)
        return Model([noise, label], img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64, input_dim=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.summary()
        img = Input(shape=self.img_shape)
        # Extract feature representation
        features = model(img)
        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        # normal_or_bot = Dense(self.num_classes, activation="softmax")(features)
        normal_or_bot = Dense(1, activation="sigmoid")(features)
        return Model(img, [validity, normal_or_bot])

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

        GEN_Validity = []
        NORMAL_Est = []
        REAL_BOT_Eva = []

        list_loss_real = []
        list_loss_fake = []
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

            XGB = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
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

            print("Estimating XGB..")
            pred_total_XGB = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                         test_Bots[data_cols], XGB, ONLY_GZ=False)

            print("Estimating DT..")
            pred_total_DT = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                        test_Bots[data_cols], DT, ONLY_GZ=False)

            print("Estimating NB..")
            pred_total_NB = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                        test_Bots[data_cols], NB, ONLY_GZ=False)

            print("Estimating RF..")
            pred_total_RF = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                        test_Bots[data_cols], RF, ONLY_GZ=False)

            print("Estimating LR..")
            pred_total_LR = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                        test_Bots[data_cols], LR, ONLY_GZ=False)

            print("Estimating KNN..")
            pred_total_KNN = predict_clf(G_Z[data_cols], test_Normal[data_cols],
                                         test_Bots[data_cols], KNN, ONLY_GZ=False)

        print("Starting GAN Training..")
        start_log_time = time.time()

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        # with tf.device(gpu_device):
        for i in range(starting_step, starting_step + nb_steps):
            for j in range(k_d):
                np.random.seed(i + j)
                z = np.random.normal(mean, stdv, (batch_size, rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(mean, stdv,
                                          (batch_size, rand_noise_dim))
                # shuffled_T_B_batch = get_data_batch(shuffled_T_B[0:int(shuffled_T_B.shape[0]*.7)], batch_size, seed=i+j)
                shuffled_T_B_batch = get_data_batch(shuffled_T_B,
                                                    batch_size,
                                                    seed=i + j)
                Labels = shuffled_T_B_batch['Label']
                sampled_labels = np.random.randint(0, 2, (batch_size, 1))
                g_z = self.generator.predict([z, sampled_labels])
                d_l_N_B = self.discriminator.train_on_batch(
                    shuffled_T_B_batch[data_cols], [real, Labels])
                d_l_g = self.discriminator.train_on_batch(
                    g_z, [fake, sampled_labels])

            for j in range(k_g):
                np.random.seed(i + j)
                z = np.random.normal(mean, stdv, (batch_size, rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(mean, stdv,
                                          (batch_size, rand_noise_dim))
                # sampled_labels = np.random.randint(0, 2, (batch_size, 1))
                g_loss = self.combined.train_on_batch([z, sampled_labels],
                                                      [real, sampled_labels])

                #       str(g_loss) + str(self.combined.metrics_names))
            list_batch_iteration.append(batch_iteration)
            # print('d_l_g: ' + str(d_l_g) + '\nd_l_N_B: ' + str(d_l_N_B)  + '\ng_loss: ' + str(g_loss))
            # ---------------------------------------------------------------------------------------------------------------
            # Determine xgb loss each step, after training generator and discriminator
            if not i % log_interval:  # 2x faster than testing each step...
                list_log_iteration.append(log_iteration)
                log_iteration = log_iteration + 1

                list_loss_real.append(d_l_N_B[0])
                list_loss_fake.append(d_l_g[0])
                list_loss_g.append(g_loss[0])

                x = get_data_batch(train, test_size, seed=i)
                z = np.random.normal(size=(test_size, rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(size=(test_size, rand_noise_dim))
                sampled_labels = np.random.randint(0, 2, (test_size, 1))
                g_z = self.generator.predict([z, sampled_labels])
                # g_z -= g_z.min()
                # g_z /= g_z.max()
                x = x[data_cols]
                # ====================Testing on 30% of train data ================================

                # print('test_Bots.shape: ' + str(test_Bots.shape))
                Test_N_or_B = pd.concat([test_Normal, test_Bots]).reset_index(
                    drop=True)  # Augmenting with real botnets
                z = np.random.normal(size=(Test_N_or_B.shape[0],
                                           rand_noise_dim))
                if USE_UNIFORM_NOISE:
                    z = np.random.uniform(size=(Test_N_or_B.shape[0],
                                                rand_noise_dim))
                Sampled_labels = np.random.randint(0, 2,
                                                   (Test_N_or_B.shape[0], 1))
                G_z = self.generator.predict([z, Sampled_labels])
                G_Z = pd.DataFrame(G_z).copy()
                G_Z.columns = data_cols
                G_Z['Label'] = 0
                pred_G_Z = self.discriminator.predict(G_Z[data_cols])
                pred_Normal = self.discriminator.predict(
                    test_Normal[data_cols])
                pred_Bots = self.discriminator.predict(test_Bots[data_cols])
                g_valid = round(sum(pred_G_Z[0]) / G_Z.shape[0], 4)

                Normal_Est = round(
                    sum(pred_Normal[1]) / test_Normal.shape[0], 4
                )  # predict normal being normal. If it maintains near 1 then it means it is normal becasue normal has been labeled as 0.
                Real_Bot_Eva = round(
                    sum(pred_Bots[1]) / test_Bots.shape[0], 4
                )  # predict evasion of real bot. If it maintains near 0 then it means it is bot because bot has been labeled as 1.
                GEN_Validity = np.append(GEN_Validity, g_valid)
                NORMAL_Est = np.append(NORMAL_Est, Normal_Est)
                REAL_BOT_Eva = np.append(REAL_BOT_Eva, Real_Bot_Eva)

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
                # ---------------------------------------------------------------------------------------------------------------
                # ---------------------------------------------------------------------------------------------------------------
                # ---------------------------------------------------------------------------------------------------------------
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

        GEN_VALIDITY = pd.DataFrame(GEN_Validity, columns=['GEN_VALIDITY'])
        NORMAL_EST = pd.DataFrame(NORMAL_Est, columns=['NORMAL_EST'])
        REAL_BOT_EVA = pd.DataFrame(REAL_BOT_Eva, columns=['REAL_BOT_EVA'])

        D_Loss_Real = pd.DataFrame(list_loss_real, columns=['D_Loss_Real'])
        D_Loss_Fake = pd.DataFrame(list_loss_fake, columns=['D_Loss_Fake'])
        G_Loss = pd.DataFrame(list_loss_g, columns=['G_Loss'])

        time_taken = pd.DataFrame([round(total_time_so_far / 60, 1)], columns=['Time'])

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
                pred_total_XGB, columns=['pred_total_XGB'])
            pred_total_DT = pd.DataFrame(
                pred_total_DT, columns=['pred_total_DT'])
            pred_total_NB = pd.DataFrame(
                pred_total_NB, columns=['pred_total_NB'])
            pred_total_RF = pd.DataFrame(
                pred_total_RF, columns=['pred_total_RF'])
            pred_total_LR = pd.DataFrame(
                pred_total_LR, columns=['pred_total_LR'])
            pred_total_KNN = pd.DataFrame(
                pred_total_KNN, columns=['pred_total_KNN'])

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
                      NORMAL_EST, REAL_BOT_EVA, D_Loss_Real, D_Loss_Fake, G_Loss, pred_total_XGB, pred_total_DT, pred_total_NB, pred_total_RF, pred_total_LR, pred_total_KNN, pred_G_Z_XGB, pred_G_Z_DT, pred_G_Z_NB, pred_G_Z_RF, pred_G_Z_LR, pred_G_Z_KNN, time_taken]
        else:
            frames = [GEN_VALIDITY,
                      NORMAL_EST, REAL_BOT_EVA, D_Loss_Real, D_Loss_Fake, G_Loss, time_taken]

        LISTS = pd.concat(frames, sort=True, axis=1).to_csv(
            CACHE_PATH + TODAY + '/ACGAN_CC_LISTS.csv')

        epoch_number = 0

        log_iteration = 0
        epoch_number = 0
        return [
            best_xgb_acc_index, best_xgb_rcl_index, best_dt_acc_index,
            best_dt_rcl_index, best_nb_acc_index, best_nb_rcl_index,
            best_rf_acc_index, best_rf_rcl_index, best_lr_acc_index,
            best_lr_rcl_index, best_knn_acc_index, best_knn_rcl_index
        ]


def train_ACGAN_CC(arguments,
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
    cache_prefix = 'ACGAN'
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
    ] = ACGAN_CC(IMG_SHAPE=len(data_cols)).train(model_components)
    return [
        best_xgb_acc_index, best_xgb_rcl_index, best_dt_acc_index,
        best_dt_rcl_index, best_nb_acc_index, best_nb_rcl_index,
        best_rf_acc_index, best_rf_rcl_index, best_lr_acc_index,
        best_lr_rcl_index, best_knn_acc_index, best_knn_rcl_index
    ]

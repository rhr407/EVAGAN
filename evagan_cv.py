from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from header import *
import header
import importlib

importlib.reload(header)  # For reloading after making changes


class EVAGAN_CV:
    def __init__(self, IMG_SHAPE=0):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 1
        self.latent_dim = 100
        # tf.compat.v1.disable_eager_execution()
        optimizer = Adam(0.0002, 0.5)
        losses = ["binary_crossentropy", "binary_crossentropy", "binary_crossentropy"]
        comb_losses = ["binary_crossentropy", "binary_crossentropy"]
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses, optimizer=optimizer)
        # Build the generator
        self.generator = self.build_generator()
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        print("label.shape: " + str(label.shape))
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
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype="int32")
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = tf.keras.layers.multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):
        model = Sequential()
        model.add(
            Conv2D(
                16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        img = Input(shape=self.img_shape)
        # Extract feature representation
        features = model(img)
        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        bot = Dense(1, activation="sigmoid")(features)
        normal = Dense(1, activation="sigmoid")(features)

        model.summary()
        return Model(img, [validity, bot, normal])

    def sample_images(self, epoch, path):
        r, c = 1, 1
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        sampled_labels = np.zeros((1, 1))
        # sampled_labels = np.random.randint(0, 1, (1, 1))

        # sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(1, 1)
        axs.imshow(gen_imgs[0, :, :, 0], cmap="gray")
        axs.axis("off")
        fig.savefig(path + "/epoch_%d.png" % epoch)
        # plt.show()
        plt.close()

    def train(self, model_components):
        [
            cache_prefix,
            with_class,
            starting_step,
            train,
            Train,
            Test,
            data_cols,
            data_dim,
            label_cols,
            label_dim,
            rand_noise_dim,
            nb_steps,
            batch_size,
            k_d,
            k_g,
            critic_pre_train_steps,
            log_interval,
            learning_rate,
            base_n_count,
            CACHE_PATH,
            FIGS_PATH,
            show,
            comb_loss,
            disc_loss_generated,
            disc_loss_real,
            xgb_losses,
            dt_losses,
            nb_losses,
            knn_losses,
            rf_losses,
            lr_losses,
            test_size,
            epoch_list_disc_loss_real,
            epoch_list_disc_loss_generated,
            epoch_list_comb_loss,
            gpu_device,
            EVALUATION_PARAMETER,
            TODAY,
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

        list_loss_real_bot = []
        list_loss_real_normal = []
        list_loss_fake_bot = []
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
        mean = 0
        stdv = 1

        if DEBUG:
            print("======================================================")
            print("Batch Size Selected -------->>>>>>> " + str(batch_size))
            print("======================================================")
        # with tf.device(gpu_device):
        # Load the dataset
        (X_train, Y_train), (_, _) = mnist.load_data()
        X_Normal, Y_Normal = (
            X_train[np.where(Y_train == 1)],
            Y_train[np.where(Y_train == 1)],
        )
        X_Bots, Y_Bots = (
            X_train[np.where(Y_train == 0)],
            Y_train[np.where(Y_train == 0)],
        )
        x_Normal = X_Normal[0 : int(X_Normal.shape[0] * 0.7)]
        y_Normal = Y_Normal[0 : int(Y_Normal.shape[0] * 0.7)]

        x_Bots = X_Bots[0 : int(X_Bots.shape[0] * 0.7 * 0.1)]  # Undersampling here
        y_Bots = Y_Bots[0 : int(Y_Bots.shape[0] * 0.7 * 0.1)]  # Undersampling here
        # y_Bots = Y_Bots[0:8]
        print("x_Normal.shape: " + str(x_Normal.shape))
        print("x_Bots.shape: " + str(x_Bots.shape))
        x_Normal = (x_Normal.astype(np.float32) - 127.5) / 127.5
        x_Normal = np.expand_dims(x_Normal, axis=3)
        # y_Normal = y_Normal.reshape(-1, 1)
        x_Bots = (x_Bots.astype(np.float32) - 127.5) / 127.5
        x_Bots = np.expand_dims(x_Bots, axis=3)
        # y_Bots = y_Bots.reshape(-1, 1)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        print("Starting GAN Training..")
        start_log_time = time.time()

        for i in range(starting_step, starting_step + nb_steps):
            idx_Normal = np.random.randint(0, x_Normal.shape[0], batch_size)
            imgs_Normal = x_Normal[idx_Normal]
            idx_Bots = np.random.randint(0, x_Bots.shape[0], batch_size)
            imgs_Bots = x_Bots[idx_Bots]
            # print('imgs_Normal.shape: ' +str(imgs_Normal.shape))
            # print('imgs_Bots.shape: ' +str(imgs_Bots.shape))
            imgs_Normal_labels = y_Normal[idx_Normal]
            imgs_Bots_labels = y_Bots[idx_Bots]
            # print('imgs_Normal_labels.shape: ' +str(imgs_Normal_labels.shape))
            # print('imgs_Bots_labels.shape: ' +str(imgs_Bots_labels.shape))
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            sampled_labels = np.zeros((batch_size, 1))

            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Train the discriminator
            d_loss_fake = self.discriminator.train_on_batch(
                gen_imgs, [fake, imgs_Bots_labels, imgs_Normal_labels]
            )
            d_loss_real_N = self.discriminator.train_on_batch(
                imgs_Normal, [real, imgs_Bots_labels, imgs_Normal_labels]
            )
            d_loss_real_B = self.discriminator.train_on_batch(
                imgs_Bots, [real, imgs_Bots_labels, imgs_Normal_labels]
            )

            d_loss = np.add(d_loss_fake, d_loss_real_N)
            d_loss = np.add(d_loss, d_loss_real_B) / 3

            g_loss = self.combined.train_on_batch(
                [noise, sampled_labels], [real, sampled_labels]
            )
            # d_loss_fake   = self.discriminator.train_on_batch(gen_imgs,    [fake, imgs_Bots_labels, imgs_Normal_labels])
            # d_loss_real_N = self.discriminator.train_on_batch(imgs_Normal, [real, imgs_Bots_labels, imgs_Normal_labels])
            # d_loss_real_B = self.discriminator.train_on_batch(imgs_Bots,   [real, imgs_Bots_labels, imgs_Normal_labels])
            # d_loss = np.add(d_loss_fake, d_loss_real_N)
            # d_loss = np.add(d_loss, d_loss_real_B) / 3
            # g_loss = self.combined.train_on_batch([noise, sampled_labels], [real, imgs_Normal_labels])
            # Plot the progress

            list_batch_iteration.append(batch_iteration)
            # ---------------------------------------------------------------------------------------------------------------
            # Determine xgb loss each step, after training generator and discriminator
            if not i % log_interval:  # 2x faster than testing each step...
                # print("[d_loss_fake:" + str(d_loss_fake))
                # print("[d_loss_real_N:" + str(d_loss_real_N))
                # print("[d_loss_real_B:" + str(d_loss_real_B))
                # print("[g_loss:" + str(g_loss))

                list_log_iteration.append(log_iteration)
                log_iteration = log_iteration + 1

                list_loss_real_bot.append(d_loss_real_B[0])
                list_loss_real_normal.append(d_loss_real_N[0])
                list_loss_fake_bot.append(d_loss_fake[0])
                list_loss_g.append(g_loss[0])
                # ====================Testing on 30% of train data ================================
                test_x_Normal = X_Normal[
                    int(X_Normal.shape[0] * 0.7) : X_Normal.shape[0]
                ]
                test_x_Bots = X_Bots[int(X_Bots.shape[0] * 0.7) : X_Bots.shape[0]]
                # Configure inputs
                test_x_Normal = (test_x_Normal.astype(np.float32) - 127.5) / 127.5
                test_x_Normal = np.expand_dims(test_x_Normal, axis=3)
                test_x_Bots = (test_x_Bots.astype(np.float32) - 127.5) / 127.5
                test_x_Bots = np.expand_dims(test_x_Bots, axis=3)

                noise = np.random.normal(0, 1, (test_x_Bots.shape[0], self.latent_dim))
                sampled_labels = np.zeros((test_x_Bots.shape[0], 1))
                gen_imgs = self.generator.predict([noise, sampled_labels])

                pred_Normal = self.discriminator.predict(test_x_Normal)
                pred_Bots = self.discriminator.predict(test_x_Bots)
                pred_Gen_Real_Fake = self.discriminator.predict(gen_imgs)
                # print('pred_Bots: ' + str(pred_Bots))
                G_Z_Real_acc_ = 0
                G_Z_N_acc_ = 0
                G_Z_B_acc_ = 0
                sum_pred_Normal = 0
                sum_pred_Bots = 0

                Ev_Gen_Bot = round(
                    sum(pred_Gen_Real_Fake[1]) / test_x_Bots.shape[0], 4
                )  #

                Gen_Real_Fake = round(
                    sum(pred_Gen_Real_Fake[0]) / test_x_Bots.shape[0], 4
                )

                N_acc_ = round(
                    sum(pred_Normal[2]) / test_x_Normal.shape[0], 4
                )  # predict normal being normal. If it maintains near 1 then it means it is normal becasue normal has been labeled as 0.
                Ev_Real_Bot = round(sum(pred_Bots[1]) / test_x_Bots.shape[0], 4)  #
                # print('N_acc_: ' + str(N_acc_))
                # print('Ev_Real_Bot: ' + str(Ev_Real_Bot))
                Disc_G_Z_Real_acc_ = np.append(Disc_G_Z_Real_acc_, Gen_Real_Fake)
                Disc_G_Z_B_acc_ = np.append(Disc_G_Z_B_acc_, Ev_Gen_Bot)
                Disc_N_acc_ = np.append(Disc_N_acc_, N_acc_)
                Disc_B_acc_ = np.append(Disc_B_acc_, Ev_Real_Bot)
                # xgb_loss = Evaluate_Parameter(x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim , classifier = 'XGB', EVALUATION_PARAMETER = EVALUATION_PARAMETER)
                # xgb_acc =  np.append(xgb_acc, xgb_loss[0])
                # xgb_rcl =  np.append(xgb_rcl, xgb_loss[1])
                # best_xgb_acc_index = list(xgb_acc).index( xgb_acc.min())
                # best_xgb_rcl_index = list(xgb_rcl).index( xgb_rcl.min())
                path = CACHE_PATH + TODAY
                self.sample_images(log_iteration, path)

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
                        print(
                            "log_iteration: "
                            + str(log_iteration)
                            + "/"
                            + str(nb_steps // log_interval)
                        )
                    # print("Time taken so far: " + str(total_time_so_far)  + " seconds")
                    total_time = (
                        total_time_so_far / log_iteration * nb_steps // log_interval
                    )
                    if DEBUG:
                        print(
                            "Average time per log_iteration: "
                            + str(total_time_so_far / log_iteration)
                        )
                    time_left = round((total_time - total_time_so_far) / 3600, 2)
                    time_unit = "hours"
                    if time_left < 1:
                        time_left = round(time_left * 60, 2)
                        time_unit = "minutes"
                    print("Time left = " + str(time_left) + " " + time_unit)
                    print(
                        "Total Time Taken: "
                        + str(round(total_time_so_far / 60, 1))
                        + " minutes"
                    )
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
                epoch_number = epoch_number + 1
                # if PLOT_AFTER_EPOCH:
                #     PlotData(
                #         x,
                #         g_z,
                #         data_cols=data_cols,
                #         label_cols=label_cols,
                #         seed=0,
                #         with_class=with_class,
                #         data_dim=data_dim,
                #         save=False,
                #         list_batch_iteration=list_batch_iteration,
                #         Disc_G_Z_Real_acc_=Disc_G_Z_Real_acc_,
                #         Disc_G_Z_N_acc_=Disc_G_Z_N_acc_,
                #         Disc_G_Z_B_acc_=Disc_G_Z_B_acc_,
                #         Disc_N_acc_=Disc_N_acc_,
                #         Disc_B_acc_=Disc_B_acc_,
                #         xgb_acc=xgb_acc,
                #         dt_acc=dt_acc,
                #         nb_acc=nb_acc,
                #         knn_acc=knn_acc,
                #         rf_acc=rf_acc,
                #         lr_acc=lr_acc,
                #         xgb_rcl=xgb_rcl,
                #         dt_rcl=dt_rcl,
                #         nb_rcl=nb_rcl,
                #         knn_rcl=knn_rcl,
                #         rf_rcl=rf_rcl,
                #         lr_rcl=lr_rcl,
                #         xgb_c2st=xgb_c2st,
                #         dt_c2st=dt_c2st,
                #         nb_c2st=nb_c2st,
                #         knn_c2st=knn_c2st,
                #         rf_c2st=rf_c2st,
                #         lr_c2st=lr_c2st,
                #         epoch_list_disc_loss_real=epoch_list_disc_loss_real,
                #         epoch_list_disc_loss_generated=epoch_list_disc_loss_generated,
                #         epoch_list_gen_loss=epoch_list_gen_loss,
                #         epoch_list_disc_acc_real=epoch_list_disc_acc_real,
                #         epoch_list_disc_acc_generated=epoch_list_disc_acc_generated,
                #         epoch_list_gen_acc=epoch_list_gen_acc,
                #         list_log_iteration=list_log_iteration,
                #         figs_path=FIGS_PATH,
                #         cache_prefix=cache_prefix,
                #         EVALUATION_PARAMETER=EVALUATION_PARAMETER,
                #         save_fig=False,
                #         GAN_type='EVAGAN_CV',
                #         TODAY=TODAY)
        epoch_number = 0
        # if DEBUG:
        #     print('Plotting...\n')
        # PlotData(x,
        #          g_z,
        #          data_cols=data_cols,
        #          label_cols=label_cols,
        #          seed=0,
        #          with_class=with_class,
        #          data_dim=data_dim,
        #          save=False,
        #          list_batch_iteration=list_batch_iteration,
        #          Disc_G_Z_Real_acc_=Disc_G_Z_Real_acc_,
        #          Disc_G_Z_N_acc_=Disc_G_Z_N_acc_,
        #          Disc_G_Z_B_acc_=Disc_G_Z_B_acc_,
        #          Disc_N_acc_=Disc_N_acc_,
        #          Disc_B_acc_=Disc_B_acc_,
        #          xgb_acc=xgb_acc,
        #          dt_acc=dt_acc,
        #          nb_acc=nb_acc,
        #          knn_acc=knn_acc,
        #          rf_acc=rf_acc,
        #          lr_acc=lr_acc,
        #          xgb_rcl=xgb_rcl,
        #          dt_rcl=dt_rcl,
        #          nb_rcl=nb_rcl,
        #          knn_rcl=knn_rcl,
        #          rf_rcl=rf_rcl,
        #          lr_rcl=lr_rcl,
        #          xgb_c2st=xgb_c2st,
        #          dt_c2st=dt_c2st,
        #          nb_c2st=nb_c2st,
        #          knn_c2st=knn_c2st,
        #          rf_c2st=rf_c2st,
        #          lr_c2st=lr_c2st,
        #          epoch_list_disc_loss_real=epoch_list_disc_loss_real,
        #          epoch_list_disc_loss_generated=epoch_list_disc_loss_generated,
        #          epoch_list_gen_loss=epoch_list_gen_loss,
        #          epoch_list_disc_acc_real=epoch_list_disc_acc_real,
        #          epoch_list_disc_acc_generated=epoch_list_disc_acc_generated,
        #          epoch_list_gen_acc=epoch_list_gen_acc,
        #          list_log_iteration=list_log_iteration,
        #          figs_path=FIGS_PATH,
        #          cache_prefix=cache_prefix,
        #          EVALUATION_PARAMETER=EVALUATION_PARAMETER,
        #          save_fig=True,
        #          GAN_type='EVAGAN_CV',
        #          TODAY=TODAY)

        log_iteration = 0
        epoch_number = 0

        Disc_G_Z_Real_acc_ = pd.DataFrame(Disc_G_Z_Real_acc_, columns=["GEN_Validity"])
        Disc_G_Z_B_acc_ = pd.DataFrame(Disc_G_Z_B_acc_, columns=["GEN_Eva"])

        Disc_N_acc_ = pd.DataFrame(Disc_N_acc_, columns=["ONE_Est"])
        Disc_B_acc_ = pd.DataFrame(Disc_B_acc_, columns=["ZERO_Eva"])

        D_Loss_ZERO_Fake = pd.DataFrame(
            list_loss_fake_bot, columns=["D_Loss_ZERO_Fake"]
        )
        D_Loss_ZERO_Real = pd.DataFrame(
            list_loss_real_bot, columns=["D_Loss_ZERO_Real"]
        )
        D_Loss_ONE_Real = pd.DataFrame(
            list_loss_real_normal, columns=["D_Loss_ONE_Real"]
        )
        G_Loss = pd.DataFrame(list_loss_g, columns=["G_Loss"])

        frames = [
            Disc_G_Z_Real_acc_,
            Disc_G_Z_B_acc_,
            Disc_N_acc_,
            Disc_B_acc_,
            D_Loss_ZERO_Fake,
            D_Loss_ZERO_Real,
            D_Loss_ONE_Real,
            G_Loss,
        ]

        LISTS = pd.concat(frames, sort=True, axis=1).to_csv(
            CACHE_PATH + TODAY + "/EVAGAN_CV_LISTS.csv"
        )

        return [
            best_xgb_acc_index,
            best_xgb_rcl_index,
            best_dt_acc_index,
            best_dt_rcl_index,
            best_nb_acc_index,
            best_nb_rcl_index,
            best_rf_acc_index,
            best_rf_rcl_index,
            best_lr_acc_index,
            best_lr_rcl_index,
            best_knn_acc_index,
            best_knn_rcl_index,
        ]


def train_EVAGAN_CV(
    arguments, train, Train, Test, data_cols, label_cols=[], seed=0, starting_step=0
):
    [
        rand_noise_dim,
        nb_steps,
        batch_size,
        k_d,
        k_g,
        critic_pre_train_steps,
        log_interval,
        learning_rate,
        base_n_count,
        CACHE_PATH,
        FIGS_PATH,
        show,
        test_size,
        gpu_device,
        EVALUATION_PARAMETER,
        TODAY,
    ] = arguments
    with_class = False
    # np.random.seed(seed)     # set random seed
    data_dim = len(data_cols)
    # print('data_dim: ', data_dim)
    # print('data_cols: ', data_cols)
    label_dim = 0
    cache_prefix = "EVAGAN_CV"
    (
        comb_loss,
        disc_loss_generated,
        disc_loss_real,
        xgb_losses,
        dt_losses,
        nb_losses,
        knn_losses,
        rf_losses,
        lr_losses,
        epoch_list_disc_loss_real,
        epoch_list_disc_loss_generated,
        epoch_list_comb_loss,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [])
    model_components = [
        cache_prefix,
        with_class,
        starting_step,
        train,
        Train,
        Test,
        data_cols,
        data_dim,
        label_cols,
        label_dim,
        rand_noise_dim,
        nb_steps,
        batch_size,
        k_d,
        k_g,
        critic_pre_train_steps,
        log_interval,
        learning_rate,
        base_n_count,
        CACHE_PATH,
        FIGS_PATH,
        show,
        comb_loss,
        disc_loss_generated,
        disc_loss_real,
        xgb_losses,
        dt_losses,
        nb_losses,
        knn_losses,
        rf_losses,
        lr_losses,
        test_size,
        epoch_list_disc_loss_real,
        epoch_list_disc_loss_generated,
        epoch_list_comb_loss,
        gpu_device,
        EVALUATION_PARAMETER,
        TODAY,
    ]
    [
        best_xgb_acc_index,
        best_xgb_rcl_index,
        best_dt_acc_index,
        best_dt_rcl_index,
        best_nb_acc_index,
        best_nb_rcl_index,
        best_rf_acc_index,
        best_rf_rcl_index,
        best_lr_acc_index,
        best_lr_rcl_index,
        best_knn_acc_index,
        best_knn_rcl_index,
    ] = EVAGAN_CV(IMG_SHAPE=len(data_cols)).train(model_components)
    return [
        best_xgb_acc_index,
        best_xgb_rcl_index,
        best_dt_acc_index,
        best_dt_rcl_index,
        best_nb_acc_index,
        best_nb_rcl_index,
        best_rf_acc_index,
        best_rf_rcl_index,
        best_lr_acc_index,
        best_lr_rcl_index,
        best_knn_acc_index,
        best_knn_rcl_index,
    ]

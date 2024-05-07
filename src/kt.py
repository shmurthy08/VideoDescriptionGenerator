class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        # Instantiate models and compile without callbacks
        ex_model = conv_lstm_extractor()
        encoder = TransformerEncoder(
            embed_dim=hp.Int('embed_dim', 256, 2048, step=128),
            num_heads=hp.Int('num_heads', 2, 5, step=1),
            drop_rate=hp.Float('drop_rate', 0.1, 0.5, step=0.1)
        )
        decoder = TransformerDecoder(
            embed_dim=hp.Int('embed_dim', 256, 2048, step=128),
            ff_dim=hp.Int('ff_dim', 512, 4096, step=1024),
            num_heads=hp.Int('num_heads', 2, 5, step=1),
            vocab_size=vocabulary_size,
            drop_rate=hp.Float('drop_rate', 0.1, 0.5, step=0.1)
        )
        model = VideoCaptioningModel(ex_model, encoder, decoder)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                LRSchedule(
                    post_warmup_learning_rate=hp.Choice('post_warmup_learning_rate', [1e-3, 1e-4, 1e-5, 1e-6]),
                    warmup_steps=num_warmup_steps,
                    total_steps=num_train_steps
                ),
                weight_decay=hp.Choice('weight_decay', [1e-3, 1e-4, 1e-5, 1e-6])
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )


tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective=kt.Objective('val_loss', "min"),
    executions_per_trial=1,
    directory='bayesian_dir',
    project_name='fixed_multilayer_hparam_tuning_acc'
)


tuner.search(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss', verbose=1),
        TensorBoard(log_dir='logs/hparam_tuning/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, write_graph=True, write_images=True)
    ],
  
)
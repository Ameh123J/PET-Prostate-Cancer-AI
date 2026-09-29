model_H2 = build_model()
# Compile
model_H2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

model_H2.summary()
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=15,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model_resnet_transfer_H2.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
    )
]
class_weight_tensor = tf.constant(
    [class_weight_dict[0], class_weight_dict[1]],
    dtype=tf.float32
)

# 7. Data pipeline (tf.data)

def add_sample_weights(x, y):
    sw = tf.gather(class_weight_tensor, tf.cast(y, tf.int32))
    return x, y, sw

data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal")])

train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
train_ds = train_ds.shuffle(len(X_tr), seed=SEED)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(add_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(add_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)


# 8. Train — Phase 1: frozen base, train the new head only
history_phase1_H2 = model_H2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
    verbose=1,
)

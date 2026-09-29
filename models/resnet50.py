def build_model(input_shape=(125, 125, 1), unfreeze_from=-30):
    inputs = layers.Input(shape=input_shape)
    # Replicate single PET channel to 3 channels (pretrained models expect RGB)
    x = layers.Concatenate()([inputs, inputs, inputs])
    # Upsample to 224x224 — ResNet's downsampling stages expect larger inputs
    x = layers.Resizing(224, 224, interpolation="bilinear")(x)
    # ResNet's own preprocessing expects 0-255 range; your data is min-max scaled to 0-1,
    # so scale up to match what the pretrained normalization expects
    x = layers.Rescaling(255.0)(x)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base_model.trainable = False  # freeze for phase 1

    x = base_model(x, training=False)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.base_model = base_model  # keep a handle for phase 2 unfreezing
    return model

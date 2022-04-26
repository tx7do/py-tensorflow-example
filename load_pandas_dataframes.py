# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe

import pandas as pd
import tensorflow as tf


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    HEART_DATA_URL = "./dataset/heart.csv"

    # pandas读取csv文件
    data = pd.read_csv(HEART_DATA_URL)

    # 查看数据的前五行
    data.head()

    data.describe()
    data.info()

    target = data.pop('target')

    dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values))

    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))

    tf.constant(data['thal'])

    train_dataset = dataset.shuffle(len(data)).batch(1)

    model = get_compiled_model()
    model.fit(train_dataset, epochs=15)


if __name__ == "__main__":
    main()

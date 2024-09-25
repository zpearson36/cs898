import tensorflow as tf

data =  tf.keras.preprocessing.image_dataset_from_directory(
        directory="data/",
        image_size=(220,220)
        )
print(data.element_spec[0].shape)

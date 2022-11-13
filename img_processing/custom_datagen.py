class CustomDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    def __init__(self):
        super().__init__()

    def preprocess(self,img):
        print("Inside preprocess")
        return img
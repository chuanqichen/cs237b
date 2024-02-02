import argparse, pdb
import numpy as np, tensorflow as tf
from utils import IMG_SIZE, LABELS, image_generator


def classify(model, test_dir):
    """
    Classifies all images in test_dir
    :param model: Model to be evaluated
    :param test_dir: Directory including the images
    :return: None
    """
    test_img_gen = image_generator.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        classes=LABELS,
        batch_size=1,
        shuffle=False,
    )

    ######### Your code starts here #########
    # Classify all images in the given folder
    # Calculate the accuracy and the number of test samples in the folder
    # test_img_gen has a list attribute filenames where you can access the
    # filename of the datapoint
    loss, accuracy = model.evaluate(test_img_gen)
    num_test = len(test_img_gen)

    for i in range(num_test):
        image, label = test_img_gen[i]
        prediction = model(image)
        prediction = tf.nn.sigmoid(prediction)
        if not np.all(tf.where(prediction>0.5, 1, 0)==label):
            print(test_img_gen.filenames[i])
    ######### Your code ends here #########

    print(f"Evaluated on {num_test} samples.")
    print(f"Accuracy: {accuracy*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_dir", type=str, default="./CS237B_HW1/Problem_2/datasets/test/")
    FLAGS, _ = parser.parse_known_args()
    model = tf.keras.models.load_model("./trained_models/trained.h5")
    classify(model, FLAGS.test_image_dir)

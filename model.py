import csv, os, random
import cv2
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import sklearn
from sklearn.model_selection import train_test_split
import math

# Import functionality from keras
import tensorflow as tf
from keras.models import Sequential, model_from_json
import keras.layers as layers
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def read_csv(path_to_csv, plot=None, root_dir=r"./my_data/IMG"):
    """
    Read data from csv file and return two lists (path to image and steering angle).
    When plot=True, plot the distribution of steering angle.
    """
    with open(path_to_csv, "r", newline='') as f:
        recorded_data = list(csv.reader(f, delimiter=',', quotechar='|'))
    
    # Get the image path and steering angle
    path_to_image = []
    steering_angle = []
    for record in recorded_data:
        path1 = os.path.join(root_dir, os.path.basename(record[0]))
        path2 = os.path.join(root_dir, os.path.basename(record[1]))
        path3 = os.path.join(root_dir, os.path.basename(record[2]))
        path_to_image.append(path1)  # Center
        path_to_image.append(path2)  # Left
        path_to_image.append(path3)  # Right
        
        angle = float(record[3])
        correction = 0.2
        steering_angle.append(angle)
        steering_angle.append(angle + correction)
        steering_angle.append(angle - correction)
    
    if plot:
        plot_distribution(steering_angle, plot)
        
    return path_to_image, steering_angle

def plot_distribution(data, path_to_plot):
    pyplot.hist(data, bins=10)
    pyplot.title('Data Distribution')
    pyplot.xlabel('Steering Angle')
    pyplot.ylabel('density')
    pyplot.savefig(path_to_plot)

def change_brightness(image):
    """
    Randomly change the brightness of image. Image use BGR color space.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    value = random.randint(-30, 30)
    value = np.uint8(value)
    v += value
    v[v > 255] = 255
    v[v < 0] = 0

    final_hsv = cv2.merge((h, s, v))
    new_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return new_image

def flip_image(image):
    """
    Flip the image horizontally
    """
    return cv2.flip( image, 1 )

def data_augmentation(images, angles, path_to_aug=r"./aug_data"):
    """
    Three steps in data augmentation:
        1. Flip all images whose steering angle is larger than 0.25
        2. Randomly distort image multiple times based on the distribution
        3. Save all images to a new directory
    """
    new_image_paths = []
    new_angles = []
    count = 0

    for img, ang in zip(images, angles):
        # Save the original image
        new_path = os.path.join(path_to_aug, "{}_{}".format(count, os.path.basename(img)))
        if random.random() <= 0.2:
            new_image_paths.append(new_path)
            new_angles.append(ang)
            cv2.imwrite(new_path, cv2.imread(img))
            count += 1

        hist, edges = np.histogram(angles)
        max_index = np.argmax(hist)
        max_value = hist[max_index]

        # Find the index
        current_index = None
        for index, value in enumerate(edges):
            if value >= ang:
                current_index = index - 1
                break
        
        duplication = max_value // (2*hist[current_index])

        original_image = cv2.imread(img)
        flipped_image = flip_image(original_image)
        new_path = os.path.join(path_to_aug, "{}_{}".format(count, os.path.basename(img)))
        if random.random() <= 0.2:
            new_image_paths.append(new_path)
            new_angles.append(-ang)
            cv2.imwrite(new_path, flipped_image)
            count += 1

        if duplication < 1:
            pass
        else:
            # Change brightness 3 times for the original image
            for _ in range(duplication):
                if random.random()<= 0.2:
                    new_path = os.path.join(path_to_aug, "{}_{}".format(count, os.path.basename(img)))
                    new_image_paths.append(new_path)
                    new_angles.append(ang)
                    cv2.imwrite(new_path, change_brightness(original_image))
                    count += 1

            # Change brightness 3 times for the flipped image
            for _ in range(duplication):
                if random.random() <= 0.2:
                    new_path = os.path.join(path_to_aug, "{}_{}".format(count, os.path.basename(img)))
                    new_image_paths.append(new_path)
                    new_angles.append(-ang)
                    cv2.imwrite(new_path, change_brightness(flipped_image))
                    count += 1
    
    return new_image_paths, new_angles

def generate_data():
    """
    Generate Data used for training the network
    """
    images, angles = read_csv(r'./my_data/driving_log.csv', plot="./plots/original_distribution.png")
    images, angles = data_augmentation(images, angles)
    with open("aug_data.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for image, angle in zip(images, angles):
            writer.writerow([image, angle]) 
    plot_distribution(angles, r"./plots/augment_distribution.png")

def rgb2yuv(x):
    import tensorflow as tf
    return tf.image.rgb_to_hsv(x)

def create_model():
    """
    Create a convolutional neural network mentioned in NVIDIA paper
    """
    model = Sequential()
    
    # The model input is image data (320x160)
    # We need to crop it
    model.add(
        layers.Cropping2D(cropping=((50,20), (0,0)), data_format="channels_last", input_shape=(160,320,3))
        )

    # Normalize the images to [0,1], a requirement of tf.image.rgb_to_hsv(x)
    model.add(layers.Lambda(lambda x: x/255.0))

    # Convert images to YUV color space
    model.add(layers.Lambda(rgb2yuv))

    # Model in NVIDIA's paper
    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(layers.Conv2D(24, (5,5), strides=(2, 2), padding='valid', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.ELU(alpha=1.0))
    model.add(layers.Conv2D(36, (5,5), strides=(2, 2), padding='valid', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.ELU(alpha=1.0))
    model.add(layers.Conv2D(48, (5,5), strides=(2, 2), padding='valid', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.ELU(alpha=1.0))
    
    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(layers.Conv2D(64, (3,3), padding='valid', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.ELU(alpha=1.0))
    model.add(layers.Conv2D(64, (3,3), padding='valid', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.ELU(alpha=1.0))

    # Add a flatten layer
    model.add(layers.Flatten())

    # Add three fully connected layers
    model.add(layers.Dense(100, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.ELU(alpha=1.0))
    model.add(layers.Dense(50, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.ELU(alpha=1.0))
    model.add(layers.Dense(10, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.ELU(alpha=1.0))
    model.add(layers.Dense(1))

    return model

def load_data(path_to_csv, num_samples=None):
    """
    Read the csv file given and return the path to images and steering angles
    """
    image_path = []
    steering_angle = []
    with open(path_to_csv, "r", newline='') as f:
        recorded_data = csv.reader(f, delimiter=',', quotechar='|')
        for line in recorded_data:
            image_path.append(line[0])
            steering_angle.append(float(line[1]))
    
    if num_samples is not None:
        image_path = image_path[:num_samples]
        steering_angle = steering_angle[:num_samples]
    
    return image_path, steering_angle

def data_generator(path_to_images, steering_angles, batch_size=128):
    """
    Data generator for training and validation
    """
    # X and Y has to be the same size
    assert len(path_to_images) == len(steering_angles)

    num_samples = len(path_to_images)
    while 1: # Loop forever so the generator never terminates
        path_to_images, steering_angles = sklearn.utils.shuffle(path_to_images, steering_angles)
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []

            path_batch = path_to_images[offset:offset+batch_size]
            angle_batch = steering_angles[offset:offset+batch_size]
            for path, angle in zip(path_batch, angle_batch):
                image = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def test_generator(path_to_images, batch_size=128):
    """
    Data generator for training and validation
    """
    num_samples = len(path_to_images)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            images = []
            path_batch = path_to_images[offset:offset+batch_size]
            for path in path_batch:
                image = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
                images.append(image)

            yield np.array(images)

def train_model(model, images, angles, batch_size, learning_rate=1e-4):
    """
    Train the model using Adam optimizer and generator
    """
    # Split training and validation
    X_train, X_test, y_train, y_test = train_test_split(images, angles, test_size=0.1)

    # Generator
    train_generator = data_generator(X_train, y_train, batch_size)
    validation_generator = data_generator(X_test, y_test, batch_size)

    # Adam optimizer
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

    # Save the best model
    checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Train the model
    model.fit_generator(
        train_generator,
        steps_per_epoch = math.ceil(len(y_train)/batch_size), 
        validation_data = validation_generator,
        validation_steps = math.ceil(len(y_test)/batch_size),
        epochs = 200, 
        verbose = 1,
        callbacks = [checkpoint]
    )

def test_model():
    """
    Create the model, train it on the first 2000 and see if it can overfits
    """
    model = create_model()
    images, angles = load_data("aug_data.csv", 2000)

    train_model(model, images, angles, 64)
    images_for_test = images[455:465]
    actual_angles = angles[455:465]
    test_data_generator = test_generator(images_for_test, 32)

    predicted_angles = model.predict_generator(test_data_generator, steps=1)
    for predict, actual in zip(predicted_angles, actual_angles):
        print (predict, actual)

def run_on_server():
    """
    Function that runs the entire process on the server
    """
    # Data augmentation
    path_to_image, steering_angle = read_csv(r'./my_data/driving_log.csv', plot=None)
    path_to_image, steering_angle = data_augmentation(path_to_image, steering_angle, path_to_aug=r"./new_data")

    # Train model
    model = create_model()
    train_model(model, path_to_image, steering_angle, 128)

if __name__ == "__main__":
    # test_model()
    run_on_server()

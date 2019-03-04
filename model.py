import csv, os, random
import cv2
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot

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
        pyplot.hist(steering_angle, bins=10, density=False)
        pyplot.title('Data Distribution')
        pyplot.xlabel('Steering Angle')
        pyplot.ylabel('density')
        pyplot.savefig(plot)
        
    return path_to_image, steering_angle

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
    return cv2.flip( img, 1 )

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
        new_image_paths.append(new_path)
        new_angles.append(ang)
        cv2.imwrite(new_path, cv2.imread(img))
        count += 1
    
if __name__ == "__main__":
    images, angles = read_csv(r'./my_data/driving_log.csv', plot="./plots/original_distribution.png")
    data_augmentation(images, angles)
    
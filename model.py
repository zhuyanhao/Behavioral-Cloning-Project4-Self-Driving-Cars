import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

def read_csv(path_to_csv, plot=None):
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
        path_to_image.append(record[0])  # Center
        path_to_image.append(record[1])  # Left
        path_to_image.append(record[2])  # Right
        
        angle = float(record[3])
        correction = 0.2
        steering_angle.append(angle)
        steering_angle.append(angle + correction)
        steering_angle.append(angle - correction)
    
    if plot:
        pyplot.hist(steering_angle, bins=10, density=True)
        pyplot.title('Data Distribution')
        pyplot.xlabel('Steering Angle')
        pyplot.ylabel('density')
        pyplot.savefig(plot)
        
    return path_to_image, steering_angle

def distort_image(path_to_image):
    """
    Randomly distort the image and return the newly generated image.
    """

def data_augmentation(images, angles, path_to_aug):
    """
    Three steps in data augmentation:
        1. Flip all images whose steering angle is larger than 0.25
        2. Randomly distort image multiple times based on the distribution
        3. Save all images to a new directory
    """
    
if __name__ == "__main__":
    images, angles = read_csv(r'./my_data/driving_log.csv', plot="./plots/original_distribution.png")
    print (len(images), len(angles))
    
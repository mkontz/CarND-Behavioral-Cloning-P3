# Data columns
#   1)  Center Image
#   2)  Left Image
#   3)  Right Image
#   4)  Steering Angle                         
#   5)  Throttle
#   6)  Brake                         
#   7)  Speed 

import csv
import cv2
import os

def flip_RGB(input):
    # flip RGB
    output = input[:, :, ::-1]
    return output
    
def mirror_left_right(input):
    # mirror left - right
    output = input[:, ::-1, :]
    return output

def loadData(Directories):
    lines = []

    print('Loading data:')    

    for dataDir in Directories:
        print('    Loading from directory: {}'.format(dataDir))
        with open(dataDir + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            cnt = 0
            for line in reader:
                # iterate through 3 files and ensure they have proper data directory
                for f in range(3):
                    # directory with image should be in same directory as corresponding driving_log.csv
     
                    line[f] = dataDir + '/IMG/' + line[f].split('/')[-1]
                lines.append(line)
            cnt += 1
                          
            print('        Successfully loaded {} sets of images'.format(cnt))
        print('    Total sets of images: {}'.format(len(list(lines))))
    
    images = []
    measurements = []
    
    delta = 0.5
    steering_delta = [0.0, delta, -delta] # center, left, right
    cnt = 0
    for line in lines:
        for i in range(3):
            # read image: center, left, right
            image = cv2.imread(line[i])
            if image is not None:     
                # convert BGR to RGB
                image = flip_RGB(image)

                # calculate steering angle including offset to account to account for left and right images
                steer = max (-1.0, min((float(line[3]) + steering_delta[i]), 1.0))

                # add image input, steering (measurement) output
                images.append(image)
                measurements.append(steer)

                # add mirror of image input and steering output
                images.append(mirror_left_right(image))
                measurements.append(-steer)
    
    print('        Number of images/measurements: {}'.format(len(images)))
    
    return images, measurements 
'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''
import matplotlib.pyplot as plt
import math
import copy

from scipy.interpolate import interp1d
from scipy.spatial import distance
from scipy.special import softmax
import numpy as np
from flask import Flask, request
from flask import render_template
import time
import json


app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    # TODO: Start sampling (12 points)
    sample_points_X, sample_points_Y = [], []
    l=len(points_X)-1
    distance = np.cumsum(np.sqrt( np.ediff1d(points_X, to_begin=0)**2 + np.ediff1d(points_Y, to_begin=0)**2 ))
    distance = distance/distance[-1] #calculating distance

    fx, fy = interp1d( distance, points_X ), interp1d( distance, points_Y ) #using interpolate and linspace to sample 100 points that are equidistant

    alpha = np.linspace(0, 1, 100)
    sample_points_X, sample_points_Y = fx(alpha), fy(alpha)
#plt.plot(points_X,points_Y, 'o-')
    #plt.plot(sample_points_X, sample_points_Y, 'or')
    #plt.axis('equal')
   # plt.show()
    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], [] #performing sampling for all  the templates as well to make it comparable to the gesture
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)
    

def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 18
    # TODO: Do pruning (12 points)
    s=len(gesture_points_X)-1
    l=len(template_sample_points_X)
    for i in range(l):
        l1=len(template_sample_points_X[i])-1
        dx=math.sqrt((gesture_points_X[0] - template_sample_points_X[i][0])**2+ (gesture_points_Y[0]-template_sample_points_Y[i][0])**2) #finding euclidean distance between the points
        dy=math.sqrt((gesture_points_X[s] - template_sample_points_X[i][l1])**2+(gesture_points_Y[s]-template_sample_points_Y[i][l1])**2)
        if(dx < threshold and dy < threshold): #if distance is below said threshold it it considered as an valid word for that template
            valid_words.append(words[i]) #appending the word to list of valid words
            valid_template_sample_points_X.append(template_sample_points_X[i]) #obtaining the valid word's points
            valid_template_sample_points_Y.append(template_sample_points_Y[i])
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    ''' # TODO: Set your own L
    L = 1
      # TODO: Calculate shape scores (12 points)
    X_mean_gesture,Y_mean_gesture=0,0 #Initializing all variables
    X_mean_template=[0]*100
    Y_mean_template=[0]*100
    mingesturex,mingesturey=0,0
    maxgesturex,maxgesturey=0,0
    maxwhgesture,scalinggesture=0,0
    wgesturex,hgesturey=0,0
    mintemplatex , mintemplatey = [0]*100,[0]*100
    maxtemplatex , maxtemplatey = [0]*100,[0]*100
    wtemplatex   , htemplatey   = [0]*100,[0]*100
    maxwhgesture1=[0]*100
    scalingtemplate = [0]*100
    sum1, sum2=[0]*100 ,[0]*100
    shape_scores = [0]*100
    shape_scores1=[0]*100
    newgesturex , newgesturey = [],[]
    count=0
    w, h = 1000, 1000
    l=len(valid_template_sample_points_X)
    newtemplatex=[]
    newtemplatey=[]
    for row in range(l):
        newtemplatex += [[0]*100]
        newtemplatey += [[0]*100]
    newgesturex = copy.deepcopy(gesture_sample_points_X) #creating a copy of the gesture sample pointslist
    newgesturey = copy.deepcopy(gesture_sample_points_Y)
    #newtemplatex=copy.deepcopy(valid_template_sample_points_X)
    #newtemplatey=copy.deepcopy(valid_template_sample_points_Y)
    for k in range(l):
        for m in range(100):
            newtemplatex[k][m]=valid_template_sample_points_X[k][m] #creating a copy of template sample points list
            newtemplatey[k][m]=valid_template_sample_points_Y[k][m]
    X_mean_gesture=sum(newgesturex)/len(newgesturex) #finding mean of the gesture
    Y_mean_gesture=sum(newgesturey)/len(newgesturey)
    for i in range(len(newgesturex)):
        newgesturex[i] -= X_mean_gesture #subtracting mean from all point in the gesture
        newgesturey[i] -= Y_mean_gesture
    mingesturex=min(newgesturex) #finding min and max of gesture
    maxgesturex=max(newgesturex)
    mingesturey=min(newgesturey)
    maxgesturey=max(newgesturey)
    wgesturex=maxgesturex-mingesturex #calculating width of gesture
    hgesturey=maxgesturey-mingesturey
    maxwhgesture=max(wgesturex,hgesturey)
    scalinggesture=L/maxwhgesture #scaling factor
    for i in range(len(newgesturex)):
        newgesturex[i]*=scalinggesture #multiplying all points in gesture with the scaling factor
        newgesturey[i]*=scalinggesture
    for k in range(l):
        for m in range(100):
            sum1[k]=sum1[k]+newtemplatex[k][m] #same process is repeated for the templates
            sum2[k]+=newtemplatey[k][m]
        Y_mean_template[k]=sum2[k]/100
        X_mean_template[k]=sum1[k]/100
    for k in range(l):
        for m in range(100):
            newtemplatex[k][m]=newtemplatex[k][m]-X_mean_template[k]
            newtemplatey[k][m]=newtemplatey[k][m]-Y_mean_template[k]
        mintemplatex[k]=min(newtemplatex[k])
        mintemplatey[k]=min(newtemplatey[k])
        maxtemplatex[k]=max(newtemplatex[k])
        maxtemplatey[k]=max(newtemplatey[k])
        wtemplatex[k]=maxtemplatex[k]-mintemplatex[k]
        htemplatey[k]=maxtemplatey[k]-mintemplatey[k]
    for k in range(l):
        if(wtemplatex[k]>htemplatey[k]):
            maxwhgesture1[k]=wtemplatex[k]
            scalingtemplate[k]=L/maxwhgesture1[k]
        else:
             maxwhgesture1[k]=htemplatey[k]
             scalingtemplate[k]=L/maxwhgesture1[k]
    for k in range(l):
        for m in range(100):
            newtemplatex[k][m]*=scalingtemplate[k]
            newtemplatey[k][m]*=scalingtemplate[k]
            shape_scores[k]+=math.sqrt((newgesturex[m] - newtemplatex[k][m])**2+ (newgesturey[m]-newtemplatey[k][m])**2)
        shape_scores[k]=shape_scores[k]/100
        
        
    return shape_scores


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.'''
   #TODO: Calculate location scores (12 points)
    start_time = time.time()
    location_scores = []
    radius = 15
    D_gesture = []
    D_temp = []
    max1,max2 = 0 , 0
    alpha = 5/100
    delta = 0
    # TODO: Calculate location scores (12 points)
    for i in range(len(valid_template_sample_points_X)):
        locsum = 0
        gesturemin = []
        templatemin = []
        for j in range(100):
            distance1 = []
            distance2 = []
            for k in range(len(valid_template_sample_points_X[i])):
                distance1.append(math.sqrt((gesture_sample_points_X[j] - valid_template_sample_points_X[i][k])**2 + (gesture_sample_points_Y[j] - valid_template_sample_points_Y[i][k])**2)) #euclidean distance between a point in gesture with every point in the template
                distance2.append(math.sqrt((valid_template_sample_points_X[i][j] - gesture_sample_points_X[k])**2 + (valid_template_sample_points_Y[i][j] - gesture_sample_points_Y[k])**2)) #euclidean distance between a point in template with every point in gesture
            gesturemin.append(min(distance1))
            templatemin.append(min(distance2))
        for j in range(len(gesturemin)):
            max1 += max(gesturemin[j]-radius,0) #applying equation5 as shown in paper
            max2 += max(templatemin[j]-radius,0)
        D_gesture.append(max1)
        D_temp.append(max2)
        for j in range(100):
            if(D_gesture[i] == 0 and D_temp[i] == 0): #calculating delta
                delta = 0
            else:
                delta = math.sqrt((gesture_sample_points_X[j] - valid_template_sample_points_X[i][j])**2 + (gesture_sample_points_Y[j] - valid_template_sample_points_Y[i][j])**2)
            locsum += (alpha * delta)
        location_scores.append(locsum)
    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.5
    # TODO: Set your own location weight
    location_coef = 0.6
    for i in range(len(shape_scores)):
       integration_scores.append(shape_coef * shape_scores[i])  #+ location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.'''
    
    best_word=""
    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)
    integration_scores = np.asarray(integration_scores)
    valid_words=np.asarray(valid_words)
    arr1inds = np.argsort(integration_scores) # using argsort to sort
    z=[x for _,x in sorted(zip(arr1inds,valid_words))] # sorting valid_words based on the indices obtainded from argsort
    for i in range(n):
        best_word+=valid_words[i]+" "
    return best_word


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    '''gesture_points_X = [gesture_points_X]
    gesture_points_Y = [gesture_points_Y]'''

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)
    

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()
   
    a=str(round((end_time - start_time) * 1000, 5))
    return '{"best_word":  "' + best_word + ' ", "elapsed_time":"' + a + 'ms"}'


if __name__ == "__main__":
    app.run()

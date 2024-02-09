import numpy as np
import matplotlib.pyplot as plt
from helper import load_dataset
import cv2


def initialize_zeros(dim):
    w, b = np.zeros((dim, 1)), 0  # take zero of weight and bias
    return w, b


def sigmoid(x):  # activation function
    y = 1.0 / (1 + np.exp(-x))
    return y


def propagation(w, b, X, Y):
    m = X.shape[1]  # how many images we have
    A = sigmoid(np.dot(w.T, X) + b)  # Y = (Wt.X+B)  (Z)
    # cost of foreword propagation / backpropagation dZ
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    cost = np.squeeze(cost) # squeeze the dimantion
    grads = {"dw": dw, "db": db}  # fetch dw by calling dw global to easily call the value/ key of dictionary
    #  gradient value

    return grads, cost


def learn(w, b, X, Y, num_iterations, learning_rate):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagation(w, b, X, Y)
        costs.append(cost)  # add an element to the end of the list/ here just transfer value to another variable

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw # update weight and bias
        b = b - learning_rate * db

        if i % 50 == 0:
            costs.append(cost)  #update cost

        if i % 50 == 0:
            print("Cost after iteration %i :  %f" % (i, cost))  # print after every 50 iterration

        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))  # output array init
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)  # output layer

    for i in range(A.shape[1]):  # number of output
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0  # predicted output
        else:
            Y_prediction[0, i] = 1
    return Y_prediction


def log_reg_model(X_train, Y_train, X_test, Y_test, num_iterations=1000, learning_rate=0.4):
    w, b = initialize_zeros(X_train.shape[0])  # total pixal in a image
    parameters, grads, costs = learn(w, b, X_train, Y_train, num_iterations, learning_rate)

    w = parameters["w"] # call from dictanary
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)  # output of a picture
    Y_prediction_train = predict(w, b, X_train)

    np.mean((Y_prediction_test - Y_test))

    # bring the accuracy by subtraction
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    print("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    # take all the model values
    d = {"costs": costs, "Y_prediction_test": Y_prediction_test, "Y_prediction_train": Y_prediction_train, "w": w,
         "b": b, "learning_rate": learning_rate, "num_iterations": num_iterations}

    return d


# ------Preprocessing----------------#
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()  # load the data set

m_train = train_set_x_orig.shape[0]  # number of train examples
m_test = test_set_x_orig.shape[0]  # number of train examples
num_px = train_set_x_orig.shape[1]  # number of pixal, image height width

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T   # 2D of all train set, colume of a single image (12288 pixals, 209 examples)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# ---Normalize data-----#
train_set_x = train_set_x_flatten / 255  # 8 bits value
test_set_x = test_set_x_flatten / 255

# --Training---#
d = log_reg_model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005)

costs = np.squeeze(d['costs'])  # make a complete array
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per fifty)')
plt.title("Learning Rate " + str(d["learning_rate"]))
plt.show()

my_image = "img6.jpg"
fname = "images/" + my_image  # image loca
image = cv2.imread(fname)  # read the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # change the orientation of array value
my_image = cv2.resize(image, (num_px, num_px)).reshape((1, num_px * num_px * 3)).T  # resize the image for the model (1, 128800) formate
img_prediction = predict(d["w"], d["b"], my_image)  # run image through the model for prediction

plt.imshow(image)  # make the image
print("The prediction is " + str(np.squeeze(img_prediction)))  # show 1 or 0 as result
plt.show()  # show the image


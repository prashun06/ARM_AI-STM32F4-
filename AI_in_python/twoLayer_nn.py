import cv2
import matplotlib.pyplot as plt

from dnn import *
from helper import load_dataset

np.random.seed(1)
train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()  # load the data set

m_train = train_x_orig.shape[0]  # number of train examples
m_test = test_x_orig.shape[0]  # number of test examples
num_px = train_x_orig.shape[1]  # number of pixal, image height width

# reshape all images......................
train_set_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1).T  # 2D of all train set, colume of a single image (12288 pixals, 209 examples)
test_set_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# ---Normalize data-----#
train_x = train_set_x_flatten / 255  # 8 bits value
test_x = test_set_x_flatten / 255

n_x = num_px * num_px * 3  # number of pixles
n_h = 6  # number hidden layer
n_y = 1  # number of output layer

layers_dims = (n_x, n_h, n_y)


def two_layer_nn_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3500):
    grads = {}
    costs = []
    m = X.shape[1]  # number of examples
    parameters = init_parameters(n_x, n_h, n_y)  # initialize the weight and bias

    W1 = parameters["W1"]  # extract all the parameters for 2 layer
    b1 = parameters["B1"]
    W2 = parameters["W2"]
    b2 = parameters["B2"]

    for i in range(0, num_iterations):
        # forward propagetion
        A1, cache1 = linear_activation_forword(X, W1, b1, 'ReLu')  # hidden layer operation
        A2, cache2 = linear_activation_forword(A1, W2, b2, 'Sigmoid')  # final layer operation

        cost = compute_cost(A2, Y)  # cost function

        # back propagetion
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))  # Output
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "Sigmoid")  # final layer elements
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "ReLu")  # hidden layer elements

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        parameters = update_parameters(parameters, grads, learning_rate)  # update the weight and bias

        W1 = parameters["W1"]
        b1 = parameters["B1"]
        W2 = parameters["W2"]
        b2 = parameters["B2"]

        if i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


parameters = two_layer_nn_model(train_x, train_y, layers_dims=(n_x, n_h, n_y), num_iterations=2500)

train_prediction = predict(train_x, train_y, parameters)
test_prediction = predict(test_x, test_y, parameters)

print("The train prediction is " + str(train_prediction))
print("The test prediction is " + str(test_prediction))


def print_wrong_predictions(classes, X, y, p):
    a = p + y  # wrong prediction
    mislabeled_indices = np.asarray(np.where(a == 1))  # wrong prediction finding
    plt.rcParams['figure.figsize'] = (50.0, 50.0)  # show which image it is
    num_images = len(mislabeled_indices[0])  # number of mislabeled images

    for i in range(num_images):
        index = mislabeled_indices[1][i]  # indexing the images
        plt.subplot(2, num_images, i + 1)  # show the images
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')  # if the display resolution is not same
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + "\n Class :" + classes[y[0, index]].decode(
                "utf-8"))


# Uncomment to show wrong predictions
# print_wrong_predictions(classes,test_x,test_y,test_prediction)
# plt.show()

my_label_y = [0]
my_image = "img6.jpg"
fname = "images/" + my_image  # image loca
image = cv2.imread(fname)  # read the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # change the orientation of array value
my_image = cv2.resize(image, (num_px, num_px)).reshape(
    (1, num_px * num_px * 3)).T  # resize the image for the model (1, 128800) formate
img_prediction = predict(my_image, my_label_y, parameters)  # run image through the model for prediction

plt.imshow(image)  # make the image
print("The prediction is " + str(np.squeeze(img_prediction)))  # show 1 or 0 as result
plt.show()  # show the image

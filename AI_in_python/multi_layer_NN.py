import matplotlib.pyplot as plt
import cv2
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


## multi layer NN model function..................................
layer_dims = [12288, 20, 8, 6, 1]  # 12288 nodes, 20 nodes in sec layer......... here 5 layers ..... need to change for NN structure

def multi_layer_nn_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3500):
    costs = []

    parameters = initialize_parameters_multi_layer(layer_dims)  # take all the parameters

    for i in range(0, num_iterations):
        AL, caches = model_forward_prop(X, parameters)  # forward propagation
        cost = compute_cost(AL, Y)  # computer the cost
        grads = model_back_prop(AL, Y, caches)  # bring gradient using backward propagation
        parameters = update_parameters(parameters, grads, learning_rate)  # update the parameters
        if i % 100 == 0:              # print result every 100 iter
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    # show the plot
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" +str(learning_rate))
    plt.show()

    return parameters


parameters = multi_layer_nn_model(train_x, train_y, layer_dims, num_iterations=2500)

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
        index = mislabeled_indices[1][i]  # indexing the images        plt.subplot(2, num_images, i + 1)  # show the images
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


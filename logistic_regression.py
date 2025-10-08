from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def predict(x, w, b):
    z = x@w + b
    probs = sigmoid(z)
    return np.where(probs >= 0.5, 1, 0)


def test_accuracy(w, b, x_test, y_test):
    y_pred = predict(x_test, w, b)
    total = len(y_test)
    correct = (y_pred == y_test).sum()
    accuracy = round(correct/total*100,2)
    return accuracy


# The iris dataset contains three classes but we'll take only two
iris = load_iris()
x = iris.data[:100]    # take all 4 features, but only 2 classes
y = iris.target[:100]  # labels 0 or 1

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize algorithm variables
n_samples, n_features = x_train.shape[0], x_train.shape[1]
w = np.zeros(n_features)
b = 0
n_epochs = 10
lr = 1e-1

for _ in range(n_epochs):
    # Predict on the tranining data
    logits = x_train@w + b
    y_pred = sigmoid(logits)

    # Calculate the gradients
    del_w = (1/n_samples)*((y_pred - y_train)).T@x_train
    del_b = (1/n_samples)*np.sum(y_pred - y_train)

    # Update the parameters
    w -= lr*del_w
    b -= lr*del_b

print(f"Finished with {n_epochs} iterations of gradient descent.\n")
print(f"The final parameters are: \n \
      w: {w} \n \
      b: {b} \n")

accuracy = test_accuracy(w, b, x_test, y_test)
print(f"The test accuracy is {accuracy}%.")
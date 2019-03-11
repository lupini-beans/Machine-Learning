import numpy as np
import matplotlib.pyplot as plt

b_history = []
w_history = []
error_history = []

def random_sample(m, b):
    np.random.seed(0)
    x = np.linspace(0,10, num=10)
    y = m*x + np.random.normal(size=10) + b

    return (x, y)

def plot_random_sample(x, y):
    plt.yticks(np.arange(50, 75, step=5))
    plt.scatter(x, y, color="purple")
    plt.title('Sample Set', fontsize=24)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.grid(True)

def loss_function(X_data, y_data):
    bb = np.arange(0,100,1)    #bias
    ww = np.arange(-5,5,0.1)   #weight
    Z = np.zeros((len(bb), len(ww)))

    for i in range(len(bb)):
        for j in range(len(ww)):
            b = bb[i]
            w = ww[j]
            Z[j][i] = 0
            for n in range(len(X_data)):
                Z[j][i] = Z[j][i] + (y_data[n] - b - w*X_data[n])**2 #loss function
            Z[j][i] = Z[j][i]/len(X_data)

    return [bb, ww, Z]

def plot_loss(bb, ww, Z):
    plt.contourf(bb, ww, Z, 50, alpha = 0.5, cmap = plt.get_cmap('jet'))
    plt.plot([50], [2], 'x', ms=12, markeredgewidth=3, color='orange') # mark out the best model
    plt.xlim(0, 100)
    plt.ylim(-5, 5)
    plt.title('Loss Landscape', fontsize=24)
    plt.xlabel(r'$b$', fontsize=16)
    plt.ylabel(r'$w$', fontsize=16)
    plt.show()

def gradient_descent(X, y, b=0.0, w=0.0, lr=0.0001, iteration=10000):
    b_history.append(b)
    w_history.append(w)


    for i in range(iteration):
        b_grad = 0.0
        w_grad = 0.0
        error = 0.0
        for n in range(len(X)):
            h = w*X[n] + b
            w_grad += X[n]*(y[n] - h)
            b_grad += y[n]-h
            error += y[n]-h

        b += (lr * b_grad)
        w += (lr * w_grad)
        b_history.append(b)
        w_history.append(w)
        error_history.append(error)

    return (w, b)

def plot_history():
    plt.title("History of Bias and Weight", fontsize=24)
    plt.xlabel("Bias History")
    plt.ylabel("Weight History")
    plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
    plt.show()

def plot_errors():
    plt.title("Errors", fontsize=24)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.plot(range(len(error_history)), error_history, color='red')
    plt.show()


def plot_mxb(X_data, y_data, w, b):
    plot_random_sample(X_data, y_data)
    x = []
    y = []
    for i in range(11):
        x.append(i)
        y.append(w*i + b)
    plt.plot(x, y, color='green')
    plt.show()


X_data, y_data = random_sample(2, 50)
print("X_data =", X_data)
print("y_data =", y_data)
plot_random_sample(X_data, y_data)
plt.show()

bb, ww, Z = loss_function(X_data, y_data)
plot_loss(bb, ww, Z)

w, b = gradient_descent(X_data, y_data)
print("Initial b = 0.0\nInitial w = 0.0\nlr = 0.0001\nIteration = 10000")
print("Generated w =", w,"\nGenerated b =", b, "\n_____________________")
plot_history()
plot_errors()


plot_mxb(X_data, y_data, w, b)
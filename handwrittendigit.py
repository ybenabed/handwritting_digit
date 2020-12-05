#!/usr/bin/env python
# coding: utf-8

# ## 1).Imports

# In[1]:


import gzip
import numpy as np 
import matplotlib.pyplot as plt


# ## 2).Read dataset

# In[2]:


def read_mnist_data(path="train-images-idx3-ubyte.gz", num_images=60000):
    dataset_file = gzip.open(path,"r")
    first_pixel_at = 16
    image_size = 28
    dataset_file.read(first_pixel_at)
    buf = dataset_file.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
    data = data.reshape(num_images, image_size, image_size, 1)
    images = np.asarray(data).squeeze()
    dataset_file.close()
    return images

def read_mnist_label(path="train-labels-idx1-ubyte.gz", num_labels=60000):
    dataset_file = gzip.open(path,"r")
    first_label_at = 8
    dataset_file.read(first_label_at)
    buf = dataset_file.read(num_labels)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    data = data.reshape(num_labels, 1)
    return data

def show_image(img):
    plt.imshow(img)
    plt.show()

index = 5467
labels = read_mnist_label()
print("Number:", labels[index])
images = read_mnist_data()
show_image(images[index])


# In[3]:


def flatten_images(images):
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
    return images.T / 255


# ## Labels encoding with OneHot 

# In[4]:


def onehot_encoding(Y):
    size = (Y.max()+1, Y.size)
    onehot = np.zeros(size)
    onehot[Y, np.arange(Y.size)] = 1
    return onehot

onehot = onehot_encoding(labels.reshape(labels.shape[0]))


# ### Load Dataset

# In[5]:


def load_dataset():
    x_train = flatten_images(read_mnist_data())
    y_train = read_mnist_label()
    y_train = onehot_encoding(y_train.reshape(y_train.shape[0]))
    x_test = flatten_images(read_mnist_data(path="t10k-images-idx3-ubyte.gz", num_images=10000))
    y_test = read_mnist_label(path="t10k-labels-idx1-ubyte.gz", num_labels=10000)
    y_test = onehot_encoding(y_test.reshape(y_test.shape[0]))
    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_dataset()


# ## 3).Initialize parameters

# In[19]:


def initialize_parameters(layer_dims):
    # We are using HE initialization
    parameters = {}
    L = len(layer_dims)
    
    np.random.seed(1)
    
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        #parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) *np.sqrt(2/(layer_dims[l-1]+layer_dims[l]))
        #parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.0001
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

def initialize_adam_parameters(parameters):
	L = len(parameters) // 2
	v = {}
	s = {}

	for l in range(L):
		v["W"+str(l+1)] = np.zeros(( parameters["W"+str(l+1)].shape[0], parameters["W"+str(l+1)].shape[1]))
		v["b"+str(l+1)] = np.zeros(( parameters["b"+str(l+1)].shape[0], parameters["b"+str(l+1)].shape[1]))
		s["W"+str(l+1)] = np.zeros(( parameters["W"+str(l+1)].shape[0], parameters["W"+str(l+1)].shape[1]))
		s["b"+str(l+1)] = np.zeros(( parameters["b"+str(l+1)].shape[0], parameters["b"+str(l+1)].shape[1]))

	return v, s


# ## Forward Propagation

# In[7]:


def linear_activation_forward(A_prev, W, b, activation="relu"):
    
    Z, linear_cache = np.dot(W, A_prev) + b, (A_prev, W, b)
    
    if activation=="relu":
        A, activation_cache = np.maximum(0, Z), np.maximum(0, Z)
        
    if activation=="softmax":
        expZ = np.exp(Z)
        A, activation_cache = expZ / np.sum(expZ, axis=0), expZ / np.sum(expZ, axis=0)
        
    if activation=="sigmoid":
        A, activation_cache = 1 / (1 + np.exp(-Z)), 1 / (1 + np.exp(-Z))
        
    cache = (linear_cache, Z)
    
    return A, cache


# In[8]:


def forward_propagation(X, parameters):
    L = len(parameters) // 2
    A = X
    caches = []
    
    for l in range(1, L):
        A_prev = A
        W, b = parameters["W"+str(l)], parameters["b"+str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "sigmoid")
        caches.append(cache)
        
    W, b = parameters["W"+str(L)], parameters["b"+str(L)]
    AL, cache = linear_activation_forward(A, W, b, activation="softmax")
    caches.append(cache)
    
    return AL, caches


# ## Compute Cost

# In[9]:


def compute_cost(AL, Y):
    J = np.mean( - np.sum( Y * np.log(AL + 1e-8), axis=0 ) )
    return J

Y_hat = np.array([ [.4, 0.2, 0.3], [0.3, 0.1, 0.2], [0.3, 0.7, .5] ])
Y = np.array([ [1., 1., 0.], [0., 0., 0.], [0., 0., 1.] ])
compute_cost(Y_hat, Y)


# ## Backward Propagation
# ### a-Derivatives
# #### RELU

# In[10]:


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


# #### Sigmoid

# In[11]:


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    return dZ


# #### Softmax

# In[12]:


def softmax_backward(A, Y):
    return A - Y


# ### b-Backward Implementation

# In[13]:


def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m
    db = np.mean(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


# In[14]:


def linear_activation_backward(dA, cache, activation="relu"):
    
    linear_cache, activation_cache = cache
    
    if activation=="relu":
        dZ = relu_backward(dA, activation_cache)
    if activation=="sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    return linear_backward(dZ, linear_cache)

def linear_activation_backward_softmax(AL, Y, cache):
    linear_cache, activation_cache = cache
    dZ = softmax_backward(AL, Y)
    return linear_backward(dZ + 1e-8, linear_cache)


# In[15]:


def backward_propagation(AL, Y, caches):
    
    derivatives = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    cache = caches[L-1]
    (derivatives["dA" + str(L-1)], derivatives["dW" + str(L)], 
     derivatives["db" + str(L)]) = linear_activation_backward_softmax(AL, Y, cache)
    
    for l in reversed(range(L-1)):
        cache = caches[l]
        (derivatives["dA" + str(l)], derivatives["dW" + str(l+1)],
         derivatives["db" + str(l+1)]) = linear_activation_backward(derivatives["dA" + str(l+1)], cache, "sigmoid")
    
    return derivatives


# ## Update Parameters

# In[68]:


def update_parameters(parameters, derivatives, learning_rate):
    
    L = len(parameters) // 2 # Here we get the number of layers
    
    for l in range(L):
        parameters["W"+str(l+1)] -= learning_rate * derivatives["dW"+str(l+1)]
        parameters["b"+str(l+1)] -= learning_rate * derivatives["db"+str(l+1)]
        
    return parameters

def update_parameters_with_adams(parameters, derivatives, v, s, t, learning_rate, beta1=0.9,
                                 beta2=0.999, epsilon=1e-8):
    
    L = len(parameters) // 2
    
    
    s_corrected = {}
    v_corrected = {}

    for l in range(L):
        v["W"+str(l+1)] = beta1 * v["W"+str(l+1)] + (1 - beta1) * derivatives["dW"+str(l+1)]
        v["b"+str(l+1)] = beta1 * v["b"+str(l+1)] + (1 - beta1) * derivatives["db"+str(l+1)]
        s["W"+str(l+1)] = beta2 * s["W"+str(l+1)] + (1 -beta2) * (derivatives["dW"+str(l+1)] ** 2)
        s["b"+str(l+1)] = beta2 * s["b"+str(l+1)] + (1 -beta2) * (derivatives["db"+str(l+1)] ** 2)
        v_corrected["W"+str(l+1)] = v["W"+str(l+1)] / (1 - beta1 ** t)
        v_corrected["b"+str(l+1)] = v["b"+str(l+1)] / (1 - beta1 ** t)
        s_corrected["W"+str(l+1)] = s["W"+str(l+1)] / (1 - beta2 ** t)
        s_corrected["b"+str(l+1)] = s["b"+str(l+1)] / (1 - beta2 ** t)
        parameters["W"+str(l+1)] -= learning_rate * v_corrected["W"+str(l+1)] / (np.sqrt(s_corrected["W"+str(l+1)]) + epsilon)
        parameters["b"+str(l+1)] -= learning_rate * v_corrected["b"+str(l+1)] / (np.sqrt(s_corrected["b"+str(l+1)]) + epsilon)

    return parameters, v, s


# ## Build the L-layers NN 

# In[65]:


def nn_model(X, Y, learning_rate=0.01, layer_dims=[28*28,40,10], num_iteration=2500, optim="none"):

    parameters = initialize_parameters(layer_dims)
    if optim=="adam":
        v, s = initialize_adam_parameters(parameters)
    for i in range(num_iteration):
        #Forward Prop
        Z3, caches = forward_propagation(X, parameters)
        #Compute the cost
        cost = compute_cost(Z3, Y)
        #BackProp
        derivatives = backward_propagation(Z3, Y, caches)
        #Update parameteres
        #parameters = update_parameters(parameters, derivatives, learning_rate)
        if optim=="none":
            parameters = update_parameters(parameters, derivatives, learning_rate)
        elif optim=="adam":
            parameters, v, s = update_parameters_with_adams(parameters,derivatives, v, s, i+1, learning_rate)
        #Print the cost
        if (i+1) % 20 == 0:
            print("Cost after",i+1,"iteration:",str(cost),"learning Rate:",learning_rate)
        if cost <= 0.02: break
        
    return parameters


# ## Make predictions 

# In[93]:


def predict(X, Y, parameters):
    Z3, cache = forward_propagation(X, parameters)
    Y_hat = np.argmax(Z3, axis=0)
    Y = np.argmax(Y, axis=0)
    accuracy = (Y_hat == Y).astype(int).mean()
    return Y, accuracy*100


# In[69]:


parameters = nn_model(x_train, y_train, learning_rate=0.01, layer_dims=[28*28,40,10], optim="adam")


# In[94]:


Y, train_accuracy = predict(x_train, y_train, parameters)
Y, test_accuracy = predict(x_test, y_test, parameters)
print("Training accuracy=",train_accuracy,"%")
print("Test accuracy=",test_accuracy,"%")


# In[ ]:





from os import listdir
from os.path import isfile, join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#------------------------------------------------------------------------------
def main():

    print("-----------Assignment 2------------")

    # importing data
    root = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(root, 'train.csv')
    test_path = os.path.join(root, 'test.csv')

    #train_data = np.genfromtxt(train_path, delimiter=',')
    #test_data = np.genfromtxt(test_path, delimiter=',')
    
    num_passes = 1000
    train = pd.read_csv(train_path)
    y = train['salary']
    train = train.drop(['id','education-num', 'salary'], axis= 1)
    train_object = train.select_dtypes(include=['object']).copy()
    train_object.head()
    train_object["sex"]= np.where(train_object["sex"].str.contains("Male"), 1, -1)
    train_object = pd.get_dummies(train_object, columns=["workclass", "education","marital-status","occupation","relationship","race","native-country"])
    train_integer = train.select_dtypes(include=['int64']).copy()
    train_integer['age'] = normalizer(train_integer['age'])
    train_integer['fnlwgt'] = normalizer(train_integer['fnlwgt'])
    #train_integer['education-num'] = normalizer(train_integer['education-num'])
    train_integer['capital-gain'] = normalizer(train_integer['capital-gain'])
    train_integer['capital-loss'] = normalizer(train_integer['capital-loss'])
    train_integer['hours-per-week'] = normalizer(train_integer['hours-per-week'])
    train_new = np.concatenate((train_object, train_integer), axis=1)
    #print("Final data\n", train_new)

    
    test1 = pd.read_csv(test_path)
    test = test1.drop(['id'], axis= 1)
    test_object = test.select_dtypes(include=['object']).copy()
    test_object.head()
    test_object["sex"]= np.where(test_object["sex"].str.contains("Male"), 1, -1)
    test_object = pd.get_dummies(test_object, columns=["workclass", "education","marital-status","occupation","relationship","race","native-country"])
    test_integer = test.select_dtypes(include=['int64']).copy()
    test_integer['age'] = normalizer(test_integer['age'])
    test_integer['fnlwgt'] = normalizer(test_integer['fnlwgt'])
    test_integer['education-num'] = normalizer(test_integer['education-num'])
    test_integer['capital-gain'] = normalizer(test_integer['capital-gain'])
    test_integer['capital-loss'] = normalizer(test_integer['capital-loss'])
    test_integer['hours-per-week'] = normalizer(test_integer['hours-per-week'])
    test_new = np.concatenate((test_object, test_integer), axis=1)
    #print("Final test data", test_new)
    #np.savetxt("version2_data.csv", train_new, delimiter=',')
    #np.savetxt("version2_data_test.csv", test_new, delimiter=',')

    # Since the feature vector is too big, reducing it's size to improve computation time.
    #Applying PCT.
    #calculating mean vector
    mean_vector_train = cal_mean_vec(train_new)
    mean_vector_test = cal_mean_vec(test_new)
    scatter_mat_train = scatter_mat(train_new, mean_vector_train)
    scatter_mat_test = scatter_mat(test_new, mean_vector_test)
    
    #Compute eigen values and eigen vectors
    eig_val_train, eig_vec_train = np.linalg.eig(scatter_mat_train)
    eig_val_test, eig_vec_test = np.linalg.eig(scatter_mat_test)
    show_eig(eig_val_train, eig_vec_train, train_new)
    show_eig(eig_val_test, eig_vec_test, test_new)
    #print("Eigen vectors eig_vec_sc", eig_vec_sc)

    # Visualize eigen vectors
    eig_visual(eig_val_train)
    eig_visual(eig_val_test)

    # Feature data after selecting 20 principal components
    feature_train = feature_data(eig_val_train, eig_vec_train)
    feature_test = feature_data(eig_val_test, eig_vec_test)

    # Transforming to get new subspace
    final_train = train_new.dot(feature_train.T)
    final_test = test_new.dot(feature_test.T)

    # Build a model with a 3-dimensional hidden layer
    model = build_model(50, final_train, y, num_passes, print_loss=False)
    y_output = predict(model, final_test)
    print("Final output \n", y_output)
    #np.savetxt("output1.csv", y_output, delimiter=',')
    test1.as_matrix()
    save_output(y_output, test1['id'])

def save_output(mat, test_id):
    a= np.array(["id"])
    b= test_id.reshape(np.shape(mat)[0], 1)
    c= np.array(["salary"])
    d= np.vstack((a,b))
    e= np.vstack((c,mat.reshape(np.shape(mat)[0], 1)))
    f= np.hstack((d,e))
    np.savetxt("predictions.csv", f,fmt='%s,%s',delimiter=',')

#--------------------------------------------------------------------
    
def feature_data(eig_val, eig_vec):
    # Make a list of (eigen value, eigen vector) tuples
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[i, :]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)  #sorting from High to low

    # visual confirmation of the decreasing list of eigenvalues
    #for i in eig_pairs:
        #print(i[0])

    #Plot shows that 20 and 40 PC's hold ~95% information.
    # Choosing 20 and 40 eigenvectors with largest eigen values
    feature_data = [] 
    for i in range(0, 20):
        feature_data.append(eig_pairs[i][1])
    feature_data = np.array(feature_data)
    return feature_data

#--------------------------------------------------------------------    

def show_eig(eig_val, eig_vec, mat):
    for i in range(len(eig_val)):
        eigvec_sc = eig_vec[i, :].reshape(np.shape(mat)[1],1).T
        #print("Eigen value {} from scatter matrix: {} ".format(i+1, eig_val[i]))
#---------------------------------------------------------------------
        
def eig_visual(eig_val):
    # visualization of eigen values
    fig = plt.figure(figsize=(8, 5))
    sing_vals = np.arange(len(eig_val)) + 1
    plt.plot(sing_vals, eig_val, linewidth=2)
    plt.xlabel('Principal component')
    plt.ylabel('Eigen value')
    plt.ylim(ymax = max(eig_val), ymin = min(eig_val))
    plt.title("Scree Plot")
    plt.show()
    
#---------------------------------------------------------------------    

def scatter_mat(mat, mean_vector):
    # Compute scatter matrix
    scatter_matrix = np.zeros((np.shape(mat)[1],np.shape(mat)[1]))
    for i in range(mat.shape[0]):
        scatter_matrix += (mat[i,:].reshape(mat.shape[1],1)-mean_vector).dot((mat[i,:].reshape(mat.shape[1],1)-mean_vector).T)
    return scatter_matrix        
#---------------------------------------------------------------------

def cal_mean_vec(mat):
    mean_vector = np.zeros(np.shape(mat)[1])
    mean_vector = np.mean(mat, axis=0)
    return mean_vector

#-------------------------------------------------------------------------------

def normalizer(mat):
        mean = np.mean(mat)
        std = np.std(mat)
        normalized = np.divide(np.subtract(mat, mean),std)

        return normalized
#--------------------------------------------------------------------    

#function to evaluate loss on the whole dataset
def calculate_loss(model, mat, y):
    #Gradient descent parameters
    #epsilon = 10 ** np.random.uniform(-6, -1) #learning rate 
    #reg_lambda = 10 ** np.random.uniform(-6, -1) #regularization strength
    epsilon = 0.0001
    reg_lambda = 0.0001
    num_examples = np.shape(mat)[0] #train_data is matrix storing input values
    
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate predictions
    z1= mat.dot(W1)+ b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2)+ b2
    exp_score = np.exp(z2)
    prob = exp_score / np.sum(exp_score, axis=1, keepdims= True)
    #Calculating the loss

    corect_logprob = -np.log(prob[range(num_examples), y])
    data_loss = np.sum(corect_logprob)
    # Add regularization term to loss
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

#--------------------------------------------------------------------

#Function to predict an output (0 or 1)
def predict(model, mat):
##    with open('weights.txt', 'rb') as f:
##        model = pickle.load(f)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    #Forward propagation
    z1 = mat.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_score = np.exp(z2)
    prob = exp_score / np.sum(exp_score, axis=1, keepdims=True)
    return np.argmax(prob, axis=1)

#-------------------------------------------------------------------

#This function learns the parameters for the NN and returns the model
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If true, print the loss every 1000 iterations

def build_model(nn_hdim, mat, y, num_passes, print_loss=False):
    
    num_examples = np.shape(mat)[0] #train_data is matrix storing input values
    nn_input_dim = np.shape(mat)[1] #input layer dimensionality
    nn_output_dim = 2 #output layer dimensionality
    
    #Gradient descent parameters
    #epsilon = 0.1 * np.random.randn() #learning rate 
    #reg_lambda = 0.1 * np.random.randn() #regularization strength
    epsilon = 0.0001
    reg_lambda = 0.0001
    
    #Initializing the parameters with random values. The parameters will be required to learn
    np.random.seed(0)
    W1 = 0.01 * np.random.randn(nn_input_dim, nn_hdim)
    b1 = np.zeros((1, nn_hdim))
    W2 = 0.01 * np.random.randn(nn_hdim, nn_output_dim)
    b2 = np.zeros((1, nn_output_dim))
    
    # At the end model will be returned.
    model = {}
    
    #Gradient descent. For every batch
    for i in range(0, num_passes):
        
        #forward propagation
        z1 = mat.dot(W1) + b1
        a1 = np.tanh(z1) 
        z2 = a1.dot(W2) + b2
        predicted_class= np.argmax(z2, axis=1)
        print("Training accuracy: %.2f" %(np.mean(predicted_class == y)))
        exp_score = np.exp(z2)
        prob = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        
        
        #backpropagation
        
        delta3 = prob
        delta3[range(0, num_examples), y] -= 1    #difference between predicted and original value y=original value
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1- np.power(a1, 2))
        dW1 = np.dot(mat.T, delta2) 
        db1 = np.sum(delta2, axis=0)
        
        #Add regularization terms (b1 and b2 are not regularized)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
        
        #Parameter update
        
        #using Adagrad
        cache1 = np.zeros(np.shape(dW1))
        cache2 = np.zeros(np.shape(db1))
        cache3 = np.zeros(np.shape(dW2))
        cache4 = np.zeros(np.shape(db2))
        eps = 0.000001
        cache1 += dW1**2
        W1 += (-epsilon) * dW1 /(np.sqrt(cache1)+eps)
        cache2 += db1**2
        b1 += (-epsilon) * db1 /(np.sqrt(cache2)+eps)
        cache3 += dW2**2
        W2 += (-epsilon) * dW2 /(np.sqrt(cache3)+eps)
        cache4 += db2**2
        b2 += (-epsilon) * db2 /(np.sqrt(cache4)+eps)
        
        #Normal gradient descent
        #W1 += (-epsilon) * dW1
        #b1 += (-epsilon) * db1 
        #W2 += (-epsilon) * dW2 
        #b2 += (-epsilon) * db2 
        
        # Introducing momentum
        #mu = 0.9
        #v = 0
        #W1 += mu * v + (-epsilon) * dW1 
        #b1 += mu * v + (-epsilon) * db1
        #W2 += mu * v + (-epsilon) * dW2
        #b2 += mu * v + (-epsilon) * db2
        
        #Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

#to store model as text file
##        with open('weights.txt', 'wb') as f:
##            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

        
        #Printing loss optionally. Expensive as it traverses over whole dataset.
        
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" %(i, calculate_loss(model, mat, y)))
            

    return model

if __name__=="__main__":
    main()

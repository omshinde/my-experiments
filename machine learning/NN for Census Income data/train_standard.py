import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():

    root = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(root, 'train.csv')
    test_path = os.path.join(root, 'test.csv')
    
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

    # Visualize eigen vectors
    eig_visual(eig_val_train)
    eig_visual(eig_val_test)

    # Feature data after selecting 20 principal components
    feature_train = feature_data(eig_val_train, eig_vec_train)
    feature_test = feature_data(eig_val_test, eig_vec_test)

    # Transforming to get new subspace
    final_train = train_new.dot(feature_train.T)
    final_test = test_new.dot(feature_test.T)

    # Parameter
    criterion = "entropy"
    n_neighbors = 15
    weights = "distance"

    test1.as_matrix()
    # Going to Decision tree classifier
    y_output_dt = decision_tree(final_train, y, final_test, criterion)
    save_output(y_output_dt, test1['id'], n=1)

    # Going for K Nearest neighbor classifier
    y_output_knn = knn(final_train, y, final_test, n_neighbors, weights)
    save_output(y_output_knn, test1['id'], n=2)

    # Going for Random Forest classifier
    y_output_rf = rand_forest(final_train, y, final_test, criterion)
    save_output(y_output_rf, test1['id'], n=3)
    

#-----------------------------------------------------------------------------
# Function to save output
def save_output(mat, test_id, n):
    a= np.array(["id"])
    b= test_id.reshape(np.shape(mat)[0], 1)
    c= np.array(["salary"])
    d= np.vstack((a,b))
    e= np.vstack((c,mat.reshape(np.shape(mat)[0], 1)))
    f= np.hstack((d,e))
    np.savetxt("predictions_%s.csv" % n, f,fmt='%s,%s',delimiter=',')

#------------------------------------------------------------------------------    

def cal_mean_vec(mat):
    mean_vector = np.zeros(np.shape(mat)[1])
    mean_vector = np.mean(mat, axis=0)
    return mean_vector

#-------------------------------------------------------------------------------    

def scatter_mat(mat, mean_vector):
    # Compute scatter matrix
    scatter_matrix = np.zeros((np.shape(mat)[1],np.shape(mat)[1]))
    for i in range(mat.shape[0]):
        scatter_matrix += (mat[i,:].reshape(mat.shape[1],1)-mean_vector).dot((mat[i,:].reshape(mat.shape[1],1)-mean_vector).T)
    return scatter_matrix        
#---------------------------------------------------------------------

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

def feature_data(eig_val, eig_vec):
    # Make a list of (eigen value, eigen vector) tuples
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[i, :]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)  #sorting from High to low

    # visual confirmation of the decreasing list
    #for i in eig_pairs:
        #print(i[0])

    #Plot shows that 20 PC's hold ~95% information.
    # Choosing 20 eigenvectors with largest eigen values
    feature_data = [] 
    for i in range(0, 20):
        feature_data.append(eig_pairs[i][1])
    feature_data = np.array(feature_data)
    return feature_data

#-------------------------------------------------------------------------------

def normalizer(mat):
        mean = np.mean(mat)
        std = np.std(mat)
        normalized = np.divide(np.subtract(mat, mean),std)
        return normalized
#-------------------------------------------------------------------------------
# Decision tree classifier
def decision_tree(mat_train, y, mat_test, criterion):
    print("----------Decision Tree classifier output: Predictions_1----------")
    clf = tree.DecisionTreeClassifier(criterion=criterion)
    clf.fit(mat_train, y)
    y_output_dt = clf.predict(mat_test)
    print("Final output using Decision tree classifier", y_output_dt)
    return y_output_dt
    
#-------------------------------------------------------------------------------
# K nearest neighbor classifier
def knn(mat_train, y, mat_test, n_neighbors, weights):
    print("----------K Nearest Neighbor classifier output: Predictions_2----------")
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights= weights)
    clf.fit(mat_train, y)
    y_output_knn = clf.predict(mat_test)
    print("Final output using K nearest neighbor classifier", y_output_knn)
    return y_output_knn

#-------------------------------------------------------------------------------
# Random forest classifier
def rand_forest(mat_train, y, mat_test, criterion):
    print("----------Random forest classifier output: Predictions_3----------")
    clf = RandomForestClassifier(n_estimators=50, criterion=criterion)
    clf.fit(mat_train, y)
    y_output_rf = clf.predict(mat_test)
    print("Final output using Random Forest classifier", y_output_rf)
    return y_output_rf

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

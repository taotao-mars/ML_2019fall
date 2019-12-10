import numpy as np
import random
from adaboost import AdaBoost
from decision_stump import DecisionStump



def save_result_split(final, name):
    ''' Save in a file the results of spliting'''

    with open('./data/' + name + ".txt", "w") as f:
        for i in range(len(final)):
            f.write(str(final[i]) + "\n")



def save_result_final(final, name):
    ''' Save in a file the final result to plot'''

    with open('./data/' + name + ".txt", "w") as f:
        for i in range(len(final)):
            f.write(str(final[i]) + "," + str(i+1) + "\n")



def split_data(percentage, num_sets):
    ''' split data for training and test sets '''

    with open('./data/' + "bupa.data", "rb") as f:
        data = [x.decode('utf8').strip() for x in f.readlines()]
        border = int(percentage*len(data))
        
        for i in range(num_sets):        
            random.shuffle(data)
            train_data = data[:border][:]
            test_data = data[border:][:]
            save_result_split(train_data, "bupa_train" + str(i))
            save_result_split(test_data, "bupa_test" + str(i))




def load_data(datafile_name):
    data = np.loadtxt(datafile_name, delimiter = ',')
    X = data[:,:-1] 
    Y = data[:,-1]
    Y[Y==2] = -1
    return X, Y
def calculate_error(T, score, Y):
    final = []
    for j in range(T):
        right, wrong = 0, 0
        dataset_for_this_T = score[j]	
        for i in range(len(dataset_for_this_T)):
            if dataset_for_this_T[i] == Y[i]:
                right += 1.0
            else:
                wrong += 1.0
        final.append(wrong/(right+wrong))
    
    return final



def main():
    classifier = AdaBoost(DecisionStump)

    num_sets = 50
    T = 100  
    percentage = 0.9 

    all_errors_train = []
    all_errors_test = []    
    aver_error_train = []
    aver_error_test = []

    split_data(percentage, num_sets)

    for i in range(num_sets):
        data_split_train = './data/bupa_train' + str(i) + ".txt"
        data_split_test = './data/' + "bupa_test" + str(i) + ".txt"
        X_train, Y_train = load_data(data_split_train)
        X_test, Y_test = load_data(data_split_test)
        
        score_train, score_test = classifier.run_adaboost(X_train, Y_train, T, X_test)
        error_train = calculate_error(T, score_train, Y_train)
        error_test = calculate_error(T, score_test, Y_test)
        all_errors_train.append(error_train)
        all_errors_test.append(error_test)
   

    # calculates the average errors
    for j in range(T):
        a_e_train = 0
        a_e_test = 0
        for i in range(num_sets):
            a_e_train += all_errors_train[i][j]
            a_e_test += all_errors_test[i][j]
            aver_error_train.append(a_e_train/num_sets)
            aver_error_test.append(a_e_test/num_sets)
  
    save_result_final(aver_error_train, 'train')
    save_result_final(aver_error_test, 'test')

    dataset_here = "./data/bupa.data" 
    X_all, Y_all = load_data(dataset_here)
    score_optional = classifier.run_adaboost(X_all, Y_all, T, None, True)
    save_result_final(score_optional, 'empirical')



if __name__ == '__main__':
    main()

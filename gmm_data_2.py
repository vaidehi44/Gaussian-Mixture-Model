import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


data_2_file1 = open('./data_2/class1.txt')
data_2_file2 = open('./data_2/class2.txt')

data_2_class1 = []
data_2_class2 = []

for line in data_2_file1:
    data=line.split(',')
    data_2_class1.append([float(data[0]), float(data[1]), 0])

for line in data_2_file2:
    data=line.split(',')
    data_2_class2.append([float(data[0]), float(data[1]), 1])


data_2 = []
data_2.extend(data_2_class1)
data_2.extend(data_2_class2)
X_train, X_test, y_train, y_test = train_test_split([[x[0], x[1]] for x in data_2], [x[2] for x in data_2], test_size=0.20, random_state=42)

    
def Initialization(data, k, dim):
    mean_vecs_1 = []
    mean_indices_1 = []
    mean_vecs_2 = []
    mean_indices_2 = []
    k_half = k//2
    mean_vecs = []
    while (len(mean_vecs_1)<k_half):
        index = random.randint(0, len(data_2_class1)-1)
        if index not in mean_indices_1:
            mean_vecs_1.append(data_2_class1[index][:2])
            mean_indices_1.append(index)
        else:
            continue
    while (len(mean_vecs_2)<(k-k_half)):
        index = random.randint(0, len(data_2_class2)-1)
        if index not in mean_indices_2:
            mean_vecs_2.append(data_2_class2[index][:2])
            mean_indices_2.append(index)
        else:
            continue
    
    mean_vecs.extend(mean_vecs_1)
    mean_vecs.extend(mean_vecs_2)
    
    cov_matrices = []
    for i in range(k):
        matrix = np.identity(dim)
        cov_matrices.append(matrix)
    
    mixture_coefficients = [(1/k) for i in range(k)]
    
    return mean_vecs, cov_matrices, mixture_coefficients


def gaussian_density(x, mean, cov, dim):
    x = np.array(x)
    mean = np.array(mean)
    matrix_mul = np.matmul((x-mean), np.linalg.inv(cov))
    matrix_mul = np.matmul(matrix_mul, (x-mean).reshape((dim,1)))
    res = np.exp(-0.5*matrix_mul)
    res = res*(1/((2*np.pi)**(dim/2)*np.sqrt(np.linalg.det(cov))))
    return res


def LogLikelihood(data, k, mean_vecs, cov_matrices, mix_coef, dim):
    log_likelihood = 0
    
    for i in range(len(data)):
        for j in range(k):
            x = data[i]
            k_val = j
            log_likelihood += mix_coef[k_val]*gaussian_density(x, mean_vecs[k_val], cov_matrices[k_val], dim)

    return log_likelihood


def Posterior_prob(k, k_val, x, mean_vecs, cov_matrs, mix_coef, dim):
    evidence = 0
    for i in range(k):
        evidence += mix_coef[i]*gaussian_density(x, mean_vecs[i], cov_matrs[i], dim)
    prior = mix_coef[k_val]*gaussian_density(x, mean_vecs[k_val], cov_matrs[k_val], dim)
    prior = prior/evidence
    return prior    
    
     
def GMM(data, test_data, test_data_true_class, k, dim):
    mean_vecs, cov_matrs, mix_coef = Initialization(data, k, dim)
        
    def no_of_points_assigned(mean_vecs, cov_matrs, mix_coef):
        assigned_k = []
        for i in range(len(data)):
            prior = Posterior_prob(k, 0, data[i], mean_vecs, cov_matrs, mix_coef, dim)
            assigned_k_val=0
            for j in range(1,k):
                temp = Posterior_prob(k, j, data[i], mean_vecs, cov_matrs, mix_coef, dim)
                if temp>prior:
                    prior=temp
                    assigned_k_val=j
            assigned_k.append(assigned_k_val)
        
        no_of_points = []
        for i in range(k):
            no_of_points.append(assigned_k.count(i))
        return no_of_points, assigned_k
    
    def get_new_means(curr_means, curr_covs, curr_mix):
        means = []
        no_of_points = no_of_points_assigned(curr_means, curr_covs, curr_mix)[0]
        
        for i in range(k):
            mean_vec_i = np.zeros(dim)
            for j in data:
                mean_vec_i += Posterior_prob(k, i, j, curr_means, curr_covs, curr_mix, dim)*np.array(j)
            mean_vec_i = mean_vec_i/no_of_points[i]
            means.append(mean_vec_i)
             
        return means
    
    def get_new_covs(curr_means, curr_covs, curr_mix):
        covs = []
        no_of_points = no_of_points_assigned(curr_means, curr_covs, curr_mix)[0]
        
        for i in range(k):
            cov_vec_i = np.zeros((dim, dim))
            for j in range(len(data)):
                x = np.array(data[j])
                mean_i = np.array(curr_means[i])
                cov_vec_i += Posterior_prob(k, i, x, curr_means, curr_covs, curr_mix, dim)*(x-mean_i).reshape((dim,1))*(x-mean_i).reshape((1,dim))
            cov_vec_i = cov_vec_i/no_of_points[i]
            covs.append(cov_vec_i)
            
        return covs
    
    def get_new_mix_coefs(curr_means, curr_covs, curr_mix):
        mix_coefs = []
        no_of_points = no_of_points_assigned(curr_means, curr_covs, curr_mix)[0]
        
        for i in range(k):
            mix_coefs.append(no_of_points[i]/len(data))
        
        return mix_coefs
    
    curr_means = mean_vecs
    curr_covs = cov_matrs
    curr_mix_coef = mix_coef
    log_likelihood = LogLikelihood(data, k, curr_means, curr_covs, curr_mix_coef, dim)
    
    for i in range(8):
        new_means = get_new_means(curr_means, curr_covs, curr_mix_coef)
        new_covs = get_new_covs(curr_means, curr_covs, curr_mix_coef)
        new_mix_coefs = get_new_mix_coefs(curr_means, curr_covs, curr_mix_coef)
        new_log_likelihood = LogLikelihood(data, k, new_means, new_covs, new_mix_coefs, dim)
        curr_means = new_means
        curr_covs = new_covs
        curr_mix_coef = new_mix_coefs
        log_likelihood = new_log_likelihood
    
    data_assignment = no_of_points_assigned(curr_means, curr_covs, curr_mix_coef)[1]
    
    plt.figure(figsize=(15,6))
    plt.suptitle("GMM Clustering on data_1 with K=%d"%(k), fontsize=22)
    plt.subplot(1,2,1)
    plt.title("Original classes", fontsize=18)
    plt.scatter([x[0] for x in data_2_class1], [x[1] for x in data_2_class1], color="darkblue", label="class 0")
    plt.scatter([x[0] for x in data_2_class2], [x[1] for x in data_2_class2], color="royalblue", label="class 1")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.title("Classes after clustering", fontsize=18)
    colors = ['teal', 'mediumaquamarine', 'lightgreen', 'yellowgreen', 'slateblue', 'orchid']
    for i in range(k):
        indices = np.where(np.array(data_assignment, dtype=object)==i)
        plt.scatter([x[0] for x in np.array(X_train)[indices]], [x[1] for x in np.array(X_train)[indices]], color=colors[i], label="class %d"%(i))
        
    plt.scatter([i[0] for i in curr_means],[i[1] for i in curr_means], color="black", label="centers")
    plt.legend()
    
    #Code for getting accuracy if k=2
    if k==2:
        assigned_k = [] #predicted classes
        for i in range(len(test_data)):
            prior = Posterior_prob(k, 0, test_data[i], curr_means, curr_covs, curr_mix_coef, dim)
            assigned_k_val = 0 
            for j in range(1,k):
                temp = Posterior_prob(k, j, test_data[i], curr_means, curr_covs, curr_mix_coef, dim)
                if temp>prior:
                    prior=temp
                    assigned_k_val=j
            assigned_k.append(assigned_k_val)
        
        confusion_mat = confusion_matrix(test_data_true_class, assigned_k)   
        accuracy = accuracy_score(test_data_true_class, assigned_k)
        print("Confusion matrix is -\n", confusion_mat)
        print("Accuracy is = ", accuracy)
         
    
GMM(X_train, X_test, y_test, 2, 2)
        
                 
            
        
        
        
        
    
    
    






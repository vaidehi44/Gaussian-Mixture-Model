import numpy as np
import random
import matplotlib.pyplot as plt


data_1_file1 = open('./data_1/class1.txt')
data_1_file2 = open('./data_1/class2.txt')

data_1_class1 = []
data_1_class2 = []

for line in data_1_file1:
    data=line.split(',')
    data_1_class1.append([float(data[0]), float(data[1])])

for line in data_1_file2:
    data=line.split(',')
    data_1_class2.append([float(data[0]), float(data[1])])

#plt.scatter([x[0] for x in data_1_class1], [x[1] for x in data_1_class1])
#plt.scatter([x[0] for x in data_1_class2], [x[1] for x in data_1_class2])

data_1 = []
data_1.extend(data_1_class1)
data_1.extend(data_1_class2)


def Initialization(data, k, dim):
    mean_vecs = []
    mean_indices = []
    while (len(mean_vecs)<k):
        index = random.randint(0, len(data)-1)
        if index not in mean_indices:
            mean_vecs.append(data[index])
            mean_indices.append(index)
        else:
            continue
    
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
    #print("matrix_mul", matrix_mul)
    res = np.exp(-0.5*matrix_mul)
    #print("res", res)
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

#mean, cov, mix_coef = Initialization(data_1, 2, 2)
#print("log likelihood for first iteration = ", LogLikelihood(data_1, 2, mean, cov, mix_coef, 2))

def Prior_prob(k, k_val, x, mean_vecs, cov_matrs, mix_coef, dim):
    evidence = 0
    for i in range(k):
        evidence += mix_coef[i]*gaussian_density(x, mean_vecs[i], cov_matrs[i], dim)
        #print(evidence, mix_coef, gaussian_density(x, mean_vecs[i], cov_matrs[i], dim))
    prior = mix_coef[k_val]*gaussian_density(x, mean_vecs[k_val], cov_matrs[k_val], dim)
    
    prior = prior/evidence
    return prior    
    
     
def GMM(data, k, dim):
    mean_vecs, cov_matrs, mix_coef = Initialization(data, k, dim)
    
    initial_log_like = LogLikelihood(data, k, mean_vecs, cov_matrs, mix_coef, dim)
    
    def no_of_points_assigned(mean_vecs, cov_matrs, mix_coef):
        assigned_k = []
        for i in range(len(data)):
            prior = Prior_prob(k, 0, data[i], mean_vecs, cov_matrs, mix_coef, dim)
            assigned_k_val=0
            for j in range(1,k):
                temp = Prior_prob(k, j, data[i], mean_vecs, cov_matrs, mix_coef, dim)
                if temp>prior:
                    prior=temp
                    assigned_k_val=j
            assigned_k.append(assigned_k_val)
        
        res = []
        for i in range(k):
            res.append(assigned_k.count(i))
        return res
    
    def new_means(curr_means, curr_covs, curr_mix):
        means = []
        no_of_points = no_of_points_assigned(curr_means, curr_covs, curr_mix)
        
        for i in range(k):
            mean_vec_i = np.zeros(dim)
            for j in data:
                mean_vec_i += Prior_prob(k, i, j, curr_means, curr_covs, curr_mix, dim)*np.array(j)
            mean_vec_i = mean_vec_i/no_of_points[i]
            means.append(mean_vec_i)
             
        return means
    
    def new_covs(curr_means, curr_covs, curr_mix):
        covs = []
        no_of_points = no_of_points_assigned(curr_means, curr_covs, curr_mix)
        
        for i in range(k):
            cov_vec_i = np.zeros((dim, dim))
            for j in range(len(data)):
                x = np.array(data[j])
                mean_i = np.array(curr_means[i])
                cov_vec_i += Prior_prob(k, i, x, curr_means, curr_covs, curr_mix, dim)*(x-mean_i).reshape((dim,1))*(x-mean_i).reshape((1,dim))
            cov_vec_i = cov_vec_i/no_of_points[i]
            covs.append(cov_vec_i)
            
        return covs
    
    def new_mix_coefs(curr_means, curr_covs, curr_mix):
        mix_coefs = []
        no_of_points = no_of_points_assigned(curr_means, curr_covs, curr_mix)
        
        for i in range(k):
            mix_coefs.append(no_of_points[i]/len(data))
        
        return mix_coefs
    
    curr_means = mean_vecs
    curr_covs = cov_matrs
    curr_mix_coef = mix_coef
    log_like = initial_log_like
    print("initialized -", curr_means, curr_covs, curr_mix_coef)
    
    for i in range(10):
        print("iteration -",i)
        means = new_means(curr_means, curr_covs, curr_mix_coef)
        covs = new_covs(curr_means, curr_covs, curr_mix_coef)
        mix_coefs = new_mix_coefs(curr_means, curr_covs, curr_mix_coef)
        log_likelihood = LogLikelihood(data, k, means, covs, mix_coefs, dim)
        print(log_likelihood)
        curr_means = means
        curr_covs = covs
        curr_mix_coef = mix_coefs
        log_like = log_likelihood
        print(curr_means, curr_covs, curr_mix_coef)
    
    
    plt.scatter([x[0] for x in data_1_class1], [x[1] for x in data_1_class1])
    plt.scatter([x[0] for x in data_1_class2], [x[1] for x in data_1_class2])
    
    plt.scatter([i[0] for i in curr_means],[i[1] for i in curr_means])
        
GMM(data_1, 4, 2)
        
        
            
        
        
        
        
    
    
    






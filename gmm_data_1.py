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


def LogLikelihood(data, k, mean_vecs, cov_matrices, mix_coef, dim):
    
    def gaussian_density(x, mean, cov):
        x = np.array(x)
        mean = np.array(mean)
        matrix_mul = np.matmul((x-mean), np.transpose(cov))
        matrix_mul = np.matmul(matrix_mul, (x-mean).reshape((dim,1)))
        res = np.exp(-0.5*matrix_mul)
        res = res*(1/((2*np.pi)**(dim/2)*np.sqrt(np.linalg.det(cov))))
        return res
    
    log_likelihood = 0
    
    for i in range(len(data)):
        for j in range(k):
            x = data[i]
            k_val = j
            log_likelihood += mix_coef[k_val]*gaussian_density(x, mean_vecs[k_val], cov_matrices[k_val])

    return log_likelihood

mean, cov, mix_coef = Initialization(data_1, 2, 2)
print("log likelihood for first iteration = ", LogLikelihood(data_1, 2, mean, cov, mix_coef, 2))

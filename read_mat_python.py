import scipy.io

def read_mat_file(file_path):
    data = scipy.io.loadmat(file_path)
    return data

file_path = 'path/to/your/file.mat'
mat_data = read_mat_file(file_path)
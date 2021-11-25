import scipy.io

if __name__ == '__main__':
    mat = scipy.io.loadmat('./data/flowers/imagelabels.mat')
    print(mat)
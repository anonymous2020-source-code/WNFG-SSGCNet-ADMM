import numpy as np

# OG
def overlook(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for k in range(samples):
        if k%100 == 0 : print(k)
        for i in range(length):
            for j in range(length):
                if(data[k][i]> data[k][j]):
                    adjmatrix[k][i][j] = 1
                else:
                    adjmatrix[k][i][j] = 0
    return adjmatrix


# WOG
def overlookg(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for k in range(samples):
        if k%100 == 0 : print(k)
        for i in range(length):
            for j in range(i+1,length):
                if(data[k][i] > data[k][j]):
                    adjmatrix[k][i][j] = (data[k][i] - data[k][j])/abs(j-i)
                    adjmatrix[k][j][i] = -1*(data[k][i] - data[k][j]) / abs(j-i)
                elif data[k][i] == data[k][j]:
                    adjmatrix[k][i][j] = adjmatrix[k][j][i] = 0
                else:
                    adjmatrix[k][i][j] = -1*(data[k][j]-data[k][i])/abs(j-i)
                    adjmatrix[k][j][i] = (data[k][j]- data[k][i])/abs(j-i)

    return adjmatrix
    



# HVG
def LPhorizontal_h(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for i in range(samples):
        for k in range(length):
            Y = np.zeros((1, 1))
            for l in range(k+1,length):
                if abs(k-l) <= 1:
                    adjmatrix[i][k][l] = 1
                    adjmatrix[i][l][k] = 1
                    if Y[0][0]<data[i][0]:
                        Y[0][0] = data[i][0]
                        Y = sorted(Y)
                elif data[i][k]>Y[0][0] and data[i][l]>Y[0][0]:
                    adjmatrix[i][k][l]=1
                    adjmatrix[i][l][k] = 1
                    Y[0][0] = data[i][l]
                    Y = sorted(Y)
    return adjmatrix


# LHVG
def LPhorizontal_lh(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for i in range(samples):
        for k in range(length):
            Y = np.zeros((1, 3))
            for l in range(k+1,length):
                if abs(k-l) <= 3:
                    adjmatrix[i][k][l] = 1
                    adjmatrix[i][l][k] = 1
                    if Y[0][0]<data[i][0]:
                        Y[0][0] = data[i][0]
                        Y = sorted(Y)
                elif data[i][k]>Y[0][0] and data[i][l]>Y[0][0]:
                    adjmatrix[i][k][l]=1
                    adjmatrix[i][l][k] = 1
                    Y[0][0] = data[i][l]
                    Y = sorted(Y)
    return adjmatrix


# VG
def LPvisibility_v(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    xlmatrix = np.zeros((samples, length,length))
    for k in range(samples):
        for i in range(length):
            for j in range(i, length):
                xlmatrix[k][i][j] = (data[k][j]-data[k][i])/abs(i-j+0.00001)
    for i in range(samples):
        for k in range(length):
            Y = np.ones((1, 1))
            Y = -1000*Y
            for l in range(k+1,length):
                if abs(k-l) <= 1:
                    adjmatrix[i][k][l]=1
                    adjmatrix[i][l][k]=1
                    if Y[0][0]<xlmatrix[i][k][l]:
                        Y[0][0] = xlmatrix[i][k][l]
                        Y = sorted(Y)
                elif Y[0][0]<xlmatrix[i][k][l]:
                    adjmatrix[i][k][l] = 1
                    adjmatrix[i][l][k] = 1
                    Y[0][0] = xlmatrix[i][k][l]
                    Y = sorted(Y)
                else:
                    adjmatrix[i][k][l]=0
                    adjmatrix[i][l][k]=0
    return adjmatrix


# LVG
def LPvisibility_lv(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    xlmatrix = np.zeros((samples, length,length))
    for k in range(samples):
        for i in range(length):
            for j in range(i,length):
                xlmatrix[k][i][j] = (data[k][j]-data[k][i])/abs(i-j+0.00001)
    for i in range(samples):
        for k in range(length):
            Y = np.ones((1, 3))
            Y = -1000*Y
            for l in range(k+1,length):
                if abs(k-l) <= 3:
                    adjmatrix[i][k][l]=1
                    adjmatrix[i][l][k]=1
                    if Y[0][0]<xlmatrix[i][k][l]:
                        Y[0][0] = xlmatrix[i][k][l]
                        Y = sorted(Y)
                elif Y[0][0]<xlmatrix[i][k][l]:
                    adjmatrix[i][k][l] = 1
                    adjmatrix[i][l][k] = 1
                    Y[0][0] = xlmatrix[i][k][l]
                    Y = sorted(Y)
                else:
                    adjmatrix[i][k][l]=0
                    adjmatrix[i][l][k]=0
    return adjmatrix

# WNFG2
def overlook_wnfg2(data):
    field = 10
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for k in range(samples):
        if k%100 == 0 : print(k)
        for i in range(1, length - field):
            for j in range(i + 1, i + 1 + field):
                if data[k][i] > data[k][j]:
                    adjmatrix[k][i][j] = (data[k][i] - data[k][j])/abs(j - i)
                    adjmatrix[k][j][i] = -1 * ((data[k][i] - data[k][j])/abs(j - i))
                elif data[k][i] < data[k][j]:
                    adjmatrix[k][i][j] = -1 * ((data[k][j] - data[k][i])/abs(j - i))
                    adjmatrix[k][j][i] = (data[k][j] - data[k][i])/abs(j - i)
                else:
                    adjmatrix[k][i][j] = 0
                    adjmatrix[k][j][i] = 0

        for i in range(length - field + 1, length):
            for j in range(length - field + 1, length):
                if data[k][i] > data[k][j]:
                    adjmatrix[k][i][j] = (data[k][i] - data[k][j])/abs(j - i)
                    adjmatrix[k][j][i] = -1 * ((data[k][i] - data[k][j])/abs(j - i))
                elif data[k][i] < data[k][j]:
                    adjmatrix[k][i][j] = -1 * ((data[k][j] - data[k][i])/abs(j - i))
                    adjmatrix[k][j][i] = (data[k][j] - data[k][i])/abs(j - i)
                else:
                    adjmatrix[k][i][j] = 0
                    adjmatrix[k][j][i] = 0
    return adjmatrix

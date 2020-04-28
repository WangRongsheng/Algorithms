import numpy as np
import matplotlib.pylab as plt
import timeit
import xlwt


def initialize_pop(m,n):
    pop = np.random.randint(0,2,(m,n))
    return pop


def value2to10(x):
    val = np.zeros(x.shape[0])
    for i in range(3):
        val = val + x[:, i] * (2 ** (2-i))
    for i in range(3, x.shape[1]):
        val = val + x[:,i]*(2**-(x.shape[1]-i-1))
    return val


def affinityFunction(x):
    #f = x + 10*np.sin(5*x) + 7*np.cos(4*x)
    #f = - x**2 + 2*x +3
    f = 1/3*x**3-5*x**2+9*x
    return f


def bubbleSort(x, y):
    for i in range(len(y)):
        for j in range(len(y)-i-1):
            if y[j] < y[j+1]:
                y[j], y[j+1] = y[j+1], y[j]
                x[j,:], x[j+1,:] = x[j+1,:], x[j,:]
    return x



def activate(pop, m, n, Fn, Ncl):
    y = affinityFunction(value2to10(pop))
    sortpop = bubbleSort(pop, y)
    for i in range(int(m*(1-Fn))):
        ca = np.tile(sortpop[i, :], (Ncl, 1))
        for j in range(1, Ncl):
            ind = np.ceil((n-1)*np.random.rand(3))
            for k in range(3):
                if ca[j, int(ind[k])] == 0:
                    ca[j, int(ind[k])] = 1
                else:
                    ca[j, int(ind[k])] = 0
        affi_ca_f = affinityFunction(value2to10(ca))
        ind = bubbleSort(ca, affi_ca_f)
        pop[i, :] = ind[0,:]

    pop[int(m*(1-Fn)):m,:] = np.random.randint(0,2,(int((m-m*(1-Fn))),n))
    return pop


def main():
    bounds = [-10, 10]
    precision = 0.0001
    n = int(np.ceil(np.log2((bounds[1]-bounds[0])/precision)))
    m = 200
    Ncl = 10
    Fn = 0.5
    caches = []
    pop = initialize_pop(m,n)
    for i in range(20):
        pop = activate(pop, m, n, Fn, Ncl)
        caches.append(value2to10(pop)[0])
    return pop, caches


def save_data(x):
    n = len(x)
    x = list(x)
    book = xlwt.Workbook()
    sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)
    for i in range(n):
        sheet.write(i, 0, x[i])
    book.save('AIA_data.xls')

pop, caches = main()
print(value2to10(pop)[0])
print(affinityFunction(value2to10(pop))[0])
elapsedtime = timeit.timeit(stmt=main, number=1)
print('Searching Time Elapsed:(S)', elapsedtime)
save_data(caches)



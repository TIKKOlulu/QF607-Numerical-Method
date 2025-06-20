import random
import numpy as np
import math
from matplotlib import pyplot as plt

##### estimate pi P10
def estimate_pi(n):
    count = 0
    for i in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            count += 1
    return 4 * count / n

print("pi from MC: ",estimate_pi(1000000))

##### calculate integral P11, P15
def fx(x):
    return x*x; 

def integralX2():
    N = 1000000
    accum , hsquare = 0, 0
    for i in range(N):
        x = np.random.uniform(0, 1)
        xsqur = fx(x)
        accum += xsqur
        hsquare += xsqur * xsqur
    result = accum/float(N)
    stderr = math.sqrt((hsquare/float(N) -result*result)/float(N))
    print("intgral using MC result/stderr: ", result, stderr)
        
integralX2()

##### calculate integral with Control Variate P18, P19
def fx(x):
    return x*x; 

def gx(x):
    return x; 

def mean_gx():
    return 0.5; 

# integral \int_0^1 x*x da , use x as control variate; for solve it analytically =1
def estimateBeta(): 
    np.random.seed(0)
    Npath = 1000
    data = np.random.uniform(0, 1, Npath)
    fValue = np.zeros(Npath)
    gValue = np.zeros(Npath)
    for i in range(1, Npath):
        fValue[i] =fx(data[i])
        gValue[i] =gx(data[i])
    varG = np.var(gValue)
    cov = np.cov(fValue, gValue)
    beta = cov[0][1]/varG
    print(varG)
    print(cov)
    return beta

def integralX2WithCV():
    N = 1000000
    accum , hsquare = 0, 0
    beta = estimateBeta()
    print("beta:", beta)
    mean_gxValue = mean_gx()
    for i in range(N):
        x = np.random.uniform(0, 1)
        fxValue = fx(x) + beta *( mean_gxValue -  gx(x))
        accum += fxValue        
        hsquare += fxValue * fxValue
    result = accum/float(N)
    stderr = math.sqrt((hsquare/float(N) - result*result)/float(N))
    print("result/stderr with CV: ", result, stderr)
    
integralX2WithCV()

##### test mid square method P27
def testMidSquareRNG():
    xs = [0] * 100
    xs[0] = 0.2372 # seed
    for i in range(1, 100):
        xs[i]=(int(xs[i-1]**2*1.0e6)%1e4)/1.0e4
    plt.scatter(range(100), xs)
    plt.show()

testMidSquareRNG()

##### test numpy random number generation  P42
# use histograms for visualizing the distribution of numerical data.
def testRNG():
    np.random.seed(0)
    data = np.random.normal(0, 1, 10000)
    # generates evenly spaced numbers over a specified range
    bins = np.linspace(math.ceil(min(data)),
                       math.floor(max(data)),
                       50)  # fixed number of bins
    plt.xlim([min(data) - 0.5, max(data) + 0.5]) #X-axis
    plt.hist(data, bins=bins) 
    plt.xlabel('variable X (50 evenly spaced bins)')
    plt.ylabel('count')
   # plt.savefig('../figs/rngNormal.eps', format = 'eps')
    plt.show()

testRNG()

##### Bronian Motion GEneration  P43
def testBrownian():
    np.random.seed(0)
    # generate 3 brownian motions for 1Y
    nBrownians = 3
    nTimeSteps = 366
    brownians = np.zeros((nBrownians, nTimeSteps))
    # each time step is 1 day, so standard deviation is sqrt(1/365.0)
    stdev = math.sqrt(1/365.0)
    for i in range(nBrownians):
        for j in range(1, nTimeSteps):
            brownians[i, j] = brownians[i, j-1] + np.random.normal(0, stdev)

    plt.title('3 brownian motions for 1Y')
    plt.plot(range(nTimeSteps), brownians[0])
    plt.plot(range(nTimeSteps), brownians[1])
    plt.plot(range(nTimeSteps), brownians[2])
   # plt.savefig('../figs/3brownians.eps', format = 'eps')
    plt.show()

testBrownian()





import numpy as np

def corr(x, w):
    """Weighted Correlation"""
    c = np.cov(x, aweights=w)
    d = np.diag(np.diag(c) ** -0.5)
    return np.dot(np.dot(d, c), d)

def get_corr(num_obs, w):
    X = np.random.randn(2, num_obs)
    return corr(X, w)

def get_2_corrs(num_obs, w1, w2):
    X = np.random.binomial(n=2, p=0.3, size=(2, num_obs))
    return corr(X, w1)[0, 1], corr(X, w2)[0, 1]

def get_corr_mean_and_var(num_runs, num_obs, w):
    vals = []
    for i in range(num_runs):
        vals.append(get_corr(num_obs, w)[0, 1])
    return np.mean(vals), np.var(vals) 
    
def get_corr_corr(num_runs, num_obs, w1, w2):
    vals1 = []
    vals2 = []
    for i in range(num_runs):
        val1, val2 = get_2_corrs(num_obs, w1, w2)
        vals1.append(val1)
        vals2.append(val2)
    vals = np.vstack((np.array(vals1), np.array(vals2)))
    covs = np.cov(vals)
    print covs
    print np.sqrt(covs)
    print np.var(vals[0]-vals[1])
    return np.corrcoef(vals)
    
def main1():
    n = 1000
    runs = 100000
    w = np.hstack((np.ones(n/2) * 6, np.ones(n/2) * 4))
    w = w / np.sum(w)
    print(np.sum(w**2))
    print get_corr_mean_and_var(runs, n, w)

def main2():
    n = 1000
    runs = 100000
    w1 = np.random.rand(n)
    w2 = np.random.rand(n)
    w1 = w1 / np.sum(w1)
    w2 = w2 / np.sum(w2)
    print(np.sum(w1**2))
    print(np.sum(w2**2))
    print(np.sum((w1-w2)**2))
    print get_corr_corr(runs, n, w1, w2)
    

if __name__=="__main__":
    main2()
    

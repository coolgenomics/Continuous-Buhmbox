"""
Buhmbox Simulation
Jie Yuan
2017

This simulates a population containing two independent phenotypes according to the
liability threshold model. The population contains three groups: those who are
controls in both phenotypes, cases in phenotype 1, and cases in phenotype 2. The
cases are combined into a single group, and the goal is to separate the observed
SNPs into their respective hidden phenotypes (e.g. by clustering a SNP x Individual
matrix).

"generate_pop_matrix()" is the code to simulate the individuals.

"greedy_search()" is the implementation of the greedy Buhmbox search - this will output
the resulting separation of the SNPs into two subsets.

"heterogeneity()" is the original Buhmbox implementation presented in Han et al.

"heterogeneity_jennerich()" is a precursor to Buhmbox in which the null matrix is not
assumed to be Identity. This is Jennerich's test for equivalence of two correlation
matrices.
"""
import pickle, sys, itertools, random
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.stats import norm, chi2, rankdata
from pprint import pprint
from generate_snps import generate_pss_model, calc_heritability
from generate_population import generate_cc_hl2, generate_direct_cases, \
                                            generate_case_matrix

def generate_pop_matrix(snp_props,num_inds=10000,pi_prop=0.5,db_snp_prop=0.5,verbose=True):
    cases1,cases2,controls = generate_case_matrix(snp_props,num_cases=num_inds,
                                                    pi_prop=pi_prop,num_controls=num_inds,
                                                    thresh=2.0,verbose=True)
    nsnps = len(snp_props)    
    cases = np.concatenate((cases1,cases2),axis=0)
    return cases, controls

def greedy_search(num_inds=10000,num_snps=100,eff_size=0.1):

    snp_props = generate_pss_model(nsnps=num_snps,eff_size=eff_size,eff_afreq=0.5)
    cases, controls = generate_pop_matrix(snp_props,num_inds=num_inds)
    
    indices = range(num_snps)
    clist = random.sample(indices,3)
    nclist = [x for x in indices if x not in clist]
    #best_sbb = heterogeneity(cases,controls,clist,snp_props)
    best_sbb,stop = heterogeneity_jennerich(cases,controls,clist,snp_props)
    count = 0
    #print clist, best_sbb, count
    converged = False
    
    clist_history = [vectorize(clist,indices)]
    sbb_history = [best_sbb]
    while not converged:
        better_soln = False
        cands = [x for x in indices if x not in clist]
        # Find the optimal SNP among candidates to add to the growing set
        for i in random.sample(cands,len(cands)):
            clist_temp = sorted(clist + [i])
            rmdr_list_temp = sorted([x for x in indices if x not in clist_temp])
            sbb,stop = heterogeneity_jennerich(cases,controls,clist_temp,snp_props)
            if stop:
                better_soln = False
                break
            print clist_temp, best_sbb, sbb
            if sbb > best_sbb:
                clist = clist_temp
                best_sbb = sbb
                count += 1
                clist_history.append(vectorize(clist,indices))
                sbb_history.append(best_sbb)
                better_soln = True
                break
        # also try removing SNPs already added to the set
        if len(clist) > 2:
            for i in random.sample(clist,len(clist)):
                mclist = [x for x in clist if x!=i]
                #sbb = heterogeneity(cases,controls,mclist,snp_props)
                sbb,stop = heterogeneity_jennerich(cases,controls,mclist,snp_props)
                if stop:
                    better_soln = False
                    break
                #print mclist, best_sbb, sbb
                if sbb > best_sbb:
                    clist = mclist
                    best_sbb = sbb
                    count -= 1
                    #print clist, sbb, best_sbb, count
                    clist_history.append(vectorize(clist,indices))
                    sbb_history.append(best_sbb)
                    better_soln = True
                    break
        if not better_soln:
            converged = True
    nclist = [x for x in indices if x not in clist]
    
    case1_sbb,stop = heterogeneity_jennerich(cases,controls,range(num_snps/2),snp_props)
    case2_sbb,stop = heterogeneity_jennerich(cases,controls,range(num_snps/2,num_snps),snp_props)

    # get percent right and wrong
    isct_case1 = set(clist).intersection(range(num_snps/2))
    isct_case2 = set(clist).intersection(range(num_snps/2,num_snps))
    pc_right = float(max(len(isct_case1),len(isct_case2)))/(num_snps/2)
    pc_wrong = float(min(len(isct_case1),len(isct_case2)))/(num_snps/2)

    print "-"*20
    print best_sbb, case1_sbb, case2_sbb, pc_right, pc_wrong
    
    # create plots of set membership and objective function over iterations
    clist_history = np.transpose(np.array(clist_history))
    img = plt.imshow(clist_history,interpolation='none')
    plt.xlabel('Iteration')
    plt.ylabel('SNP index (%s PSS1, %s PSS2)'%(num_snps/2,num_snps/2))
    plt.figure()
    plt.plot(sbb_history)
    plt.xlabel('Iteration')
    plt.ylabel('SBB correlation score')
    plt.show()
    
    return best_sbb, case1_sbb, case2_sbb, pc_right, pc_wrong

def vectorize(clist,indices):
    return [1 if i in clist else 0 for i in indices]
    
def buhmbox(cases,controls,clist,snp_props):
    """
    cases, controls: Numpy array where each row is an indiv and each col is a snp
    clist: tuple of indices that are the snps for DB
    snp_props: 
    """
    num_snps = len(clist)
    snp_cases = cases[:,clist]
    snp_controls = controls[:,clist]

    #R_expected = expected_corr(snp_props,clist)

    N = float(len(snp_cases))
    Np = float(len(snp_controls))
    R = np.corrcoef(snp_cases.T)
    Rp = np.corrcoef(snp_controls.T)
    Y = np.sqrt(N*Np/(N+Np)) * (R-Rp)
    #Y = np.sqrt(N*Np/(N+Np)) * (R-R_expected)
    
    pi_cases = []
    pi_controls = []
    gamma_i = []
    for i in range(num_snps):
        pi_case = np.sum(snp_cases[:,i]) / (2*snp_cases.shape[0])
        pi_cases.append(pi_case)
        pi_control = np.sum(snp_controls[:,i]) / (2*snp_controls.shape[0])
        pi_controls.append(pi_control)
        gamma_i.append( pi_case/(1-pi_case) / (pi_control/(1-pi_control)) )
    
    # calculate SBB
    numer = 0.0
    denom = 0.0
    for i in range(num_snps):
        for j in range(i+1,num_snps):
            wij = np.sqrt(pi_controls[i]*(1-pi_controls[i])*pi_controls[j]*(1-pi_controls[j])) \
                    * (gamma_i[i]-1) * (gamma_i[j] - 1) \
                    / (pi_controls[i]*(gamma_i[i]-1) + 1) / (pi_controls[j]*(gamma_i[j]-1) + 1)
            yij = Y[i,j]
            numer += wij * yij
            denom += wij * wij
    if not denom > 0.0:
        print "error: denominator 0"      
        print num_snps
        for i in range(num_snps):
            print gamma_i[i]
            for j in range(i+1,num_snps):
                wij = np.sqrt(pi_controls[i]*(1-pi_controls[i])*pi_controls[j]*(1-pi_controls[j])) \
                        * (gamma_i[i]-1) * (gamma_i[j] - 1) \
                        / (pi_controls[i]*(gamma_i[i]-1) + 1) / (pi_controls[j]*(gamma_i[j]-1) + 1)
                print wij
        sys.exit(1)

    SBB = numer / np.sqrt(denom)
    return SBB

def buhmbox_vectorized(cases,controls,clist,snp_props):
    """
    cases, controls: Numpy array where each row is an indiv and each col is a snp
    clist: tuple of indices that are the snps for DB
    snp_props: 
    """
    num_snps = len(clist)
    snp_cases = cases[:,clist]
    snp_controls = controls[:,clist]

    N = float(len(snp_cases))
    Np = float(len(snp_controls))
    R = np.corrcoef(snp_cases.T)
    Rp = np.corrcoef(snp_controls.T)
    Y = np.sqrt(N*Np/(N+Np)) * (R-Rp)
    
    pi_cases = np.sum(snp_cases, axis=0) / (2*snp_cases.shape[0])
    pi_controls = np.sum(snp_controls, axis=0) / (2*snp_controls.shape[0])
    gamma = pi_cases/(1-pi_cases) / (pi_controls/(1-pi_controls))
    
    # calculate SBB
    elem1 = np.sqrt(pi_controls*(1-pi_controls))
    elem2 = gamma-1
    elem3 = elem2 * pi_controls + 1
    mat1 = np.sqrt(np.dot(elem1.reshape((num_snps, 1)), elem1.reshape((1, num_snps))))
    mat2 = np.dot(elem2.reshape((num_snps, 1)), elem2.reshape((1, num_snps)))
    mat3 = np.dot(elem3.reshape((num_snps, 1)), elem3.reshape((1, num_snps)))
    w = mat1 * mat2 / mat3
    SBB = np.sum(np.triu(w*Y, k=1)) / np.sqrt(np.sum(np.triu(w ** 2, k=1)))
    return SBB

def get_weights(phenos):
    percentiles = (rankdata(phenos) - 1) / len(phenos)
    weights = -np.log(1 - percentiles)
    return weights / np.sum(weights)

def corr(x, w):
    """Weighted Correlation"""
    c = np.cov(x, aweights=w)
    d = np.diag(np.diag(c) ** -0.5)
    return np.dot(np.dot(d, c), d)

def continuous_buhmbox(genos, phenos, clist,snp_props):
    """
    cases, controls: Numpy array where each row is an indiv and each col is a snp
    clist: tuple of indices that are the snps for DB
    snp_props: 
    """
    num_snps = len(clist)
    snp_indivs = genos[:,clist]
    weights = get_weights(phenos)
    
    N = len(genos)
    w2 = np.sum(weights ** 2)
    R = corr(snp_indivs.T, weights)
    Rp = np.corrcoef(snp_indivs.T)
    Y = np.sqrt(1/(1/N + w2)) * (R-Rp)
    #Y = np.sqrt(N*Np/(N+Np)) * (R-R_expected)
    
    pi_pluses = []
    pi_minuses = []
    gamma_i = []
    for i in range(num_snps):
        pi_plus = np.sum(snp_indivs[:,i] * weights) / 2
        pi_minus = np.sum(snp_indivs[:,i]) / (2*snp_indivs.shape[0])
        pi_pluses.append(pi_plus)
        pi_minuses.append(pi_minus)
        gamma_i.append( pi_plus/(1-pi_plus) / (pi_minus/(1-pi_minus)) )
    
    # calculate SBB
    numer = 0.0
    denom = 0.0
    for i in range(num_snps):
        for j in range(i+1,num_snps):
            wij = np.sqrt(pi_minuses[i]*(1-pi_minuses[i])*pi_minuses[j]*(1-pi_minuses[j])) \
                    * (gamma_i[i]-1) * (gamma_i[j] - 1) \
                    / (pi_minuses[i]*(gamma_i[i]-1) + 1) / (pi_minuses[j]*(gamma_i[j]-1) + 1)
            yij = Y[i,j]
            numer += wij * yij
            denom += wij * wij
    if not denom > 0.0:
        print "error: denominator 0"      
        print num_snps
        for i in range(num_snps):
            print gamma_i[i]
            for j in range(i+1,num_snps):
                wij = np.sqrt(pi_minuses[i]*(1-pi_minuses[i])*pi_minuses[j]*(1-pi_minuses[j])) \
                        * (gamma_i[i]-1) * (gamma_i[j] - 1) \
                        / (pi_minuses[i]*(gamma_i[i]-1) + 1) / (pi_minuses[j]*(gamma_i[j]-1) + 1)
                print wij
        sys.exit(1)

    SBB = numer / np.sqrt(denom)
    return SBB

def continuous_buhmbox_vect(genos, phenos, clist,snp_props):
    """
    cases, controls: Numpy array where each row is an indiv and each col is a snp
    clist: tuple of indices that are the snps for DB
    snp_props: 
    """
    num_snps = len(clist)
    snp_indivs = genos[:,clist]
    num_indivs = snp_indivs.shape[0]
    weights = get_weights(phenos)
    
    N = len(genos)
    w2 = np.sum(weights ** 2)
    R = corr(snp_indivs.T, weights)
    Rp = np.corrcoef(snp_indivs.T)
    Y = np.sqrt(1/(1/N + w2)) * (R-Rp)
    
    pi_plus = np.sum(snp_indivs * weights.reshape((num_indivs, 1)), axis=0) / 2
    pi_minus = np.sum(snp_indivs, axis=0) / (2*num_indivs)
    gamma = pi_plus/(1-pi_plus) / (pi_minus/(1-pi_minus))
    
    # calculate SBB
    elem1 = np.sqrt(pi_minus*(1-pi_minus))
    elem2 = gamma-1
    elem3 = elem2 * pi_minus + 1
    mat1 = np.sqrt(np.dot(elem1.reshape((num_snps, 1)), elem1.reshape((1, num_snps))))
    mat2 = np.dot(elem2.reshape((num_snps, 1)), elem2.reshape((1, num_snps)))
    mat3 = np.dot(elem3.reshape((num_snps, 1)), elem3.reshape((1, num_snps)))
    w = mat1 * mat2 / mat3
    SBB = np.sum(np.triu(w*Y, k=1)) / np.sqrt(np.sum(np.triu(w ** 2, k=1)))
    return SBB

def heterogeneity(cases,controls,clist,snp_props):
    print("hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
    print cases
    print controls
    print clist
    print snp_props
    num_snps = len(clist)
    snp_cases = cases[:,clist]
    snp_controls = controls[:,clist]

    R_expected = expected_corr(snp_props,clist)

    N = float(len(snp_cases))
    Np = float(len(snp_controls))
    R = np.corrcoef(snp_cases.T)
    Rp = np.corrcoef(snp_controls.T)
    #Y = np.sqrt(N*Np/(N+Np)) * (R-Rp)
    Y = np.sqrt(N*Np/(N+Np)) * (R-R_expected)
    
    pi_cases = []
    pi_controls = []
    gamma_i = []
    for i in range(num_snps):
        pi_case = np.sum(snp_cases[:,i]) / (2*snp_cases.shape[0])
        pi_cases.append(pi_case)
        pi_control = np.sum(snp_controls[:,i]) / (2*snp_controls.shape[0])
        pi_controls.append(pi_control)
        gamma_i.append( pi_case/(1-pi_case) / (pi_control/(1-pi_control)) )
    
    # calculate SBB
    numer = 0.0
    denom = 0.0
    for i in range(num_snps):
        for j in range(i+1,num_snps):
            wij = np.sqrt(pi_controls[i]*(1-pi_controls[i])*pi_controls[j]*(1-pi_controls[j])) \
                    * (gamma_i[i]-1) * (gamma_i[j] - 1) \
                    / (pi_controls[i]*(gamma_i[i]-1) + 1) / (pi_controls[j]*(gamma_i[j]-1) + 1)
            yij = Y[i,j]
            numer += wij * yij
            denom += wij * wij
    if not denom > 0.0:
        print "error: denominator 0"      
        print num_snps
        for i in range(num_snps):
            print gamma_i[i]
            for j in range(i+1,num_snps):
                wij = np.sqrt(pi_controls[i]*(1-pi_controls[i])*pi_controls[j]*(1-pi_controls[j])) \
                        * (gamma_i[i]-1) * (gamma_i[j] - 1) \
                        / (pi_controls[i]*(gamma_i[i]-1) + 1) / (pi_controls[j]*(gamma_i[j]-1) + 1)
                print wij
        sys.exit(1)

    SBB = numer / np.sqrt(denom)
    return SBB

def heterogeneity_jennerich(cases,controls,clist,snp_props):
    print("hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
    print cases
    print controls
    print clist
    print snp_props

    # Remove correlation from SNPs not in the current set,
    # but include correlations between from in-SNP and out-SNP
    N = float(len(cases))
    R = np.corrcoef(cases.T)
    for i in range(len(snp_props)):
        for j in range(i+1,len(snp_props)):
            if i not in clist and j not in clist:
                R[i,j] = 0.0
                R[j,i] = 0.0

    # build the null matrix P (no heterogeneity)
    P = expected_corr(snp_props,clist)
    Pinv = inv(P)
    krnckr_delta = np.eye(P.shape[0])
    T = krnckr_delta + np.multiply(P,Pinv)
    Y = np.sqrt(N)*(R-P)

    # build Jennerich test statistic
    dgpinvy = Pinv.dot(Y).diagonal()
    ypyp = Y.dot(Pinv).dot(Y).dot(Pinv)
    Sjenn = 0.5*np.trace(ypyp) - (dgpinvy.transpose()).dot(inv(T)).dot(dgpinvy)

    num_snps = len(clist)
    q = num_snps*(num_snps-1.0)/2.0
    prob = chi2.pdf(Sjenn, df=q)
    if prob == 0.0:
        het_score = Sjenn
        return het_score, True
    else:
        het_score = -np.log(prob)
    return het_score, False # stop, if probability is zero

def expected_corr(snp_props, clist, T=2.0):
    """
        expected correlation in the absence of heterogeneity
        for now, assume there are always 50% case1, 50% case2
    """
    betas = np.array([s.beta for i,s in enumerate(snp_props)],ndmin=2)

    x = np.array([[0.0,1.0,2.0]])
    x2 = np.array([[0.0,1.0,4.0]])

    # replace this with pi
    betas = np.dot(betas,np.array([[0.5],[0.5]]))

    ps = np.array([s.p for i,s in enumerate(snp_props)],ndmin=2).T
    psmtrx = np.concatenate((np.multiply(1.0-ps,1.0-ps),
                             2.0*np.multiply(ps,1.0-ps),
                             np.multiply(ps,ps)),
                             axis=1)
    
    xbetas = np.dot(betas,x)
    inner_cdf = np.divide(T - xbetas,np.sqrt(1.0 - np.multiply(
                                                                np.square(betas),
                                                                np.array(psmtrx[:,1],ndmin=2).T
                                                              )))

    postrs = (1.0 - norm.cdf(inner_cdf))
    postrs = np.multiply(postrs,psmtrx)
    postrs = postrs/postrs.sum(axis=1)[:,None]

    x = np.tile(x,(len(snp_props),1))
    x2 = np.tile(x2,(len(snp_props),1))
    x_psmtrx_ps = np.multiply(x,postrs)
    x2_psmtrx_ps = np.multiply(x2,postrs)

    ex = np.sum(x_psmtrx_ps,axis=1)
    ex2 = np.sum(x2_psmtrx_ps,axis=1)

    R = np.zeros((len(snp_props),len(snp_props)))
    # calculate exy
    for xi in range(0,len(snp_props)):
        R[xi,xi] = 1.0
        for yi in range(xi+1,len(clist)):
            if xi in clist and yi in clist:
                px = np.tile(psmtrx[xi,:],(3,1)).T
                py = np.tile(psmtrx[yi,:],(3,1))

                pxpy = np.multiply(px,py)

                a = np.array([[0.0,1.0,2.0]])
                x = np.tile(a,(3,1)).T
                y = np.tile(a,(3,1))
                xy = np.multiply(x,y)
                betax = np.tile(xbetas[xi],(3,1)).T
                betay = np.tile(xbetas[yi],(3,1))

                postrs = 1.0 - norm.cdf((T-betax-betay)/np.sqrt(1.0 - betax[1,1]**2 *px[1,1] - betay[1,1]**2 *py[1,1]))
                postrs = np.multiply(postrs,pxpy)
                postrs = postrs/np.sum(postrs)

                exiyi = np.sum(
                            np.multiply(xy,postrs))
                R[xi,yi] = (exiyi - ex[xi]*ex[yi])/np.sqrt(ex2[xi]-ex[xi]**2)/np.sqrt(ex2[yi]-ex[yi]**2)
                R[yi,xi] = R[xi,yi]
    return R

def greedy_search_power_eval():
    """
        For a range of parameter values, does the greedy search identify the ground truth SNP cluster, and
        does the ground truth cluster have the highest SBB
    """    
    num_trials = 10
    num_inds_set = [1000,3000,5000,7000,10000]
    num_snps_set = [20, 40, 60, 80, 100]
    
    #best_sbb,case1_sbb,case2_sbb,pc_right,pc_wrong
    nind_means = np.empty((len(num_inds_set),5))
    for i,ninds in enumerate(num_inds_set):
        ave_res = np.empty((num_trials,5))
        for nt in range(num_trials):
            res = greedy_search(num_inds=ninds,num_snps=50)
            ave_res[nt,:] = res
        means = ave_res.mean(0)
        nind_means[i,:] = means

    nsnps_means = np.empty((len(num_snps_set),5))
    for i,nsnps in enumerate(num_snps_set):
        ave_res = np.empty((num_trials,5))
        for nt in range(num_trials):
            res = greedy_search(num_inds=10000,num_snps=nsnps)
            ave_res[nt,:] = res
        means = ave_res.mean(0)
        nsnps_means[i,:] = means

    plt.plot(num_inds_set,nind_means[:,0],label="greedy SBB")
    plt.plot(num_inds_set,nind_means[:,1],label="case1 SBB")
    plt.plot(num_inds_set,nind_means[:,2],label="case2 SBB")
    plt.xlabel("Number of individuals")
    plt.ylabel("SBB")
    plt.legend()

    plt.figure()
    plt.plot(num_inds_set,nind_means[:,3],label="percent SNPs right")
    plt.plot(num_inds_set,nind_means[:,4],label="percent SNPs wrong")
    plt.xlabel("Number of individuals")
    plt.ylabel("Percent")
    plt.legend()

    plt.figure()
    plt.plot(num_snps_set,nsnps_means[:,0],label="greedy SBB")
    plt.plot(num_snps_set,nsnps_means[:,1],label="case1 SBB")
    plt.plot(num_snps_set,nsnps_means[:,2],label="case2 SBB")
    plt.xlabel("Number of SNPs")
    plt.ylabel("SBB")
    plt.legend()

    plt.figure()
    plt.plot(num_snps_set,nsnps_means[:,3],label="percent SNPs right")
    plt.plot(num_snps_set,nsnps_means[:,4],label="percent SNPs wrong")
    plt.xlabel("Number of SNPs")
    plt.ylabel("Percent")
    plt.legend()
    plt.show()

def main():
    res = greedy_search(num_inds=10000,num_snps=20,eff_size=0.2)

if __name__=="__main__":
    main()

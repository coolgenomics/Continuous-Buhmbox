"""
Liability Threshold Model population sampler
Jie Yuan

This contains two methods of sampling genotypes of cases and controls:
- randomly generating individuals according to effect allele frequency, and
    then picking out those who are cases (whose sum(x*beta) exceeds a threshold)
- directly generating cases according to the posterior distribution of each SNP
    given the individual is a case
"""
import math, random, sys, timeit
import numpy as np
from pprint import pprint

from scipy.stats import norm
from sklearn.preprocessing import normalize

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from generate_snps_09022017 import generate_pss_model, calc_heritability

class Individual(object):
    def __init__(self,snp_props,sample_posterior=False,case1=False,case2=False,thres=2.0):
        assert len(snp_props)>0
        self.geno = [] # {0,1,2}
        #self.geno_mean0 = [] # adjusted to mean 0
        self.pheno = [0.0] * len(snp_props[0].beta)
        self.is_alt = False
        #self.pheno_sum = [0.0] * len(snp_props[0].beta)

        # sample SNPs according to SNP props
        for snp in snp_props:
            p = snp.p          
            pp = random.random()
            if pp < p**2.0:
                g = 2.0
            elif pp < p**2.0 + 2.0*p*(1.0-p):
                g = 1.0
            else:
                g = 0.0
            self.geno.append(g)
            #gg = g - 2*p
            #gg /= math.sqrt(2*p*(1-p))
            #self.geno_mean0.append(gg)

            # create phenotypes
            for i in range(len(self.pheno)):
                # self.pheno[i] += gg * snp.beta[i]
                # self.pheno_sum[i] += gg * snp.beta[i]
                self.pheno[i] += g * snp.beta[i]
                #self.pheno_sum[i] += g * snp.beta[i]
        # add error term
        """
        h0sqrd,h1sqrd = calc_heritability(snp_props)
        e0 = np.random.normal(loc=0.0,scale=math.sqrt(1-h0sqrd))
        e1 = np.random.normal(loc=0.0,scale=math.sqrt(1-h1sqrd))
        self.pheno = [x+y for x,y in zip(self.pheno,[e0,e1])]
        """

class IndividualHetero(object):
    def __init__(self,snp_props,sample_posterior=False,case1=False,case2=False,thres=2.0):
        assert len(snp_props)>0
        self.geno = [] # {0,1,2}
        #self.geno_mean0 = [] # adjusted to mean 0
        self.pheno = [0.0] * len(snp_props[0].beta)
        alt_pheno_2 = 0.0
        self.is_alt = False
        #self.pheno_sum = [0.0] * len(snp_props[0].beta)

        # sample SNPs according to SNP props
        for snp in snp_props:
            p = snp.p          
            pp = random.random()
            if pp < p**2.0:
                g = 2.0
            elif pp < p**2.0 + 2.0*p*(1.0-p):
                g = 1.0
            else:
                g = 0.0
            self.geno.append(g)
            #gg = g - 2*p
            #gg /= math.sqrt(2*p*(1-p))
            #self.geno_mean0.append(gg)

            # create phenotypes
            self.pheno[0] += g * snp.beta[0]
            self.pheno[1] += g * snp.beta[1]
            alt_pheno_2 += g * (snp.beta[0] * np.random.normal(loc=1.0))
        
        if alt_pheno_2 > self.pheno[1]:
            self.pheno[1] = alt_pheno_2
            self.is_alt = True
        # add error term
        """
        h0sqrd,h1sqrd = calc_heritability(snp_props)
        e0 = np.random.normal(loc=0.0,scale=math.sqrt(1-h0sqrd))
        e1 = np.random.normal(loc=0.0,scale=math.sqrt(1-h1sqrd))
        self.pheno = [x+y for x,y in zip(self.pheno,[e0,e1])]
        """

def generate_population_hetero(snp_props, num_inds=10000):
    inds = []
    for i in range(int(num_inds)):
        inds.append(IndividualHetero(snp_props))
    return inds
    

def generate_population(snp_props, num_inds=10000):
    inds = []
    for i in range(int(num_inds)):
        inds.append(Individual(snp_props))
    return inds

def alex_plot():
    snp_props_independent = generate_pss_model(num_snps=[100, 0, 0, 100])
    independent_pop = generate_population(snp_props_independent)
    plot_inds(independent_pop)
    
    snp_props_pleiotropic = generate_pss_model(num_snps=[50, 100, 0, 50])
    pleiotropic_pop = generate_population(snp_props_pleiotropic)
    plot_inds(pleiotropic_pop)
    
    hetero_pop = generate_population_hetero(snp_props_independent)
    plot_inds(hetero_pop)    
    
def plot_inds(inds):
    alts = filter(lambda i: i.is_alt, inds)
    normies = filter(lambda i: not i.is_alt, inds)
    if len(alts) > 0:
        plot_inds_helper(alts, "r.")
    plot_inds_helper(normies, "b.")
    plt.show()
    plot_inds_helper(inds, "b.")
    plt.show()
    
def plot_inds_helper(inds, label):
    phenos = map(lambda i: i.pheno, inds)
    phen1, phen2 = zip(*phenos)
    plt.plot(phen1, phen2, label)
    


def generate_cc_hl2(snp_props,num_cases=15000,pi_prop=0.5,num_controls=15000,thresh=2.0,verbose=True):
    """
        hL^2 is defined by SNPs, not manually specified
    """
    num_cases1 = round(num_cases*pi_prop)
    num_cases2 = round(num_cases*(1.0-pi_prop))
    print "Num cases: ", num_cases1, num_cases2

    casesPSS1 = []
    casesPSS2 = []
    controls = []
    h1sq,h2sq = calc_heritability(snp_props)
    count = 0
    while True:
        count += 1
        if verbose and count % 1 == 0:
            print "controls: ",len(controls)," casesPSS1: ",len(casesPSS1), \
                                        " casesPSS2: ",len(casesPSS2)
        inds = generate_population(snp_props, num_inds=5000)
        for ind in inds:
            ind.pheno[0] += np.random.normal(loc=0.0,scale=np.sqrt(1-h1sq))
            ind.pheno[1] += np.random.normal(loc=0.0,scale=np.sqrt(1-h2sq))

        mean1 = np.mean([x.pheno[0] for x in inds])
        std1 = np.std([x.pheno[0] for x in inds])
        mean2 = np.mean([x.pheno[1] for x in inds])
        std2 = np.std([x.pheno[1] for x in inds])

        for ind in inds:
            if (ind.pheno[0] - mean1)/std1 > thresh and len(casesPSS1) < num_cases1:
                casesPSS1.append(ind)
            elif (ind.pheno[1] - mean2)/std2 > thresh and len(casesPSS2) < num_cases2:
                casesPSS2.append(ind)
            elif len(controls) < num_controls:
                controls.append(ind)
            
            if len(casesPSS1) >= num_cases1 and \
               len (casesPSS2) >= num_cases1 and \
               len(controls) >= num_controls:
                return casesPSS1, casesPSS2, controls

def generate_direct_cases(snp_props,num_cases=20000,pi_prop=0.5,num_controls=15000,thresh=2.0,verbose=True):
    controls = []
    cases1 = []
    cases2 = []
    for i in range(num_controls):
        controls.append(Individual(snp_props,sample_posterior=True,case1=False,case2=False))
    for i in range(int(round(pi_prop*num_cases))):
        cases1.append(Individual(snp_props,sample_posterior=True,case1=True,case2=False))
    for i in range(int(round((1.0-pi_prop)*num_cases))):
        cases2.append(Individual(snp_props,sample_posterior=True,case1=False,case2=True))
    return cases1, cases2, controls

def generate_case_matrix(snp_props,num_cases=20000,pi_prop=0.5,num_controls=10000,thresh=2.0,verbose=True):
    """
        return a matrix where rows = individuals and columns = SNPs
        hL^2 is precomputed for a fixed order of SNPs
    """

    # generate controls randomly based on effect allele frequencies
    controls = np.empty([num_controls, len(snp_props)])
    eafs = [s.p for s in snp_props]
    for i,p in enumerate(eafs):
        sprobs = [(1-p)*(1-p), 2*p*(1-p), p*p]
        controls[:,i] = np.random.choice(3,size=num_controls,p=sprobs)
    
    # generate case1/case2 cases according to running posterior probabilities
    num_case1s = int(num_cases*pi_prop)
    num_case2s = int(num_cases*(1.0-pi_prop))
    
    cases1 = make_case_group(snp_props, num_case1s, case_idx=0, thresh=2.0)
    cases2 = make_case_group(snp_props, num_case2s, case_idx=1, thresh=2.0)
    return cases1, cases2, controls

def make_case_group(snp_props,num_cases,case_idx,thresh=2.0):
    """
        case_idx: index of the phenotype (PSS1, or PSS2)
    """
    cases = np.empty([num_cases, len(snp_props)])
    neff_snp_idxs = [i for i,s in enumerate(snp_props) if s.beta[case_idx] == 0.0]
    eff_snp_idxs = [i for i,s in enumerate(snp_props) if s.beta[case_idx] != 0.0]
    for idx in neff_snp_idxs:
        p = snp_props[idx].p
        sprobs = [(1-p)*(1-p), 2*p*(1-p), p*p]
        cases[:,idx] = np.random.choice(3,size=num_cases,p=sprobs)

    current_sums = np.zeros(num_cases)
    hLsq = 0.0
    for idx in eff_snp_idxs:
        beta = snp_props[idx].beta[case_idx]
        p = snp_props[idx].p
        sprobs = [(1-p)*(1-p), 2.0*p*(1-p), p*p]
        hLsq += beta**2 * 2.0*p*(1-p)
        
        gs = np.tile(np.array([0.0,1.0,2.0]), (num_cases,1))
        # dims are num_cases * [0,1,2], posterior for each individual and each possible g
        prob_mat = (np.tile(current_sums, (3,1)).T + gs*beta - thresh) / np.sqrt(1-hLsq)
        prob_mat = norm.cdf(prob_mat)
        prob_mat = np.dot(prob_mat, np.diag(sprobs))
        prob_mat = normalize(prob_mat, axis=1, norm='l1')
        prob_calc = np.cumsum(prob_mat,axis=1)[:,:2]

        # sample from probability matrix (each row will have different probs for [0,1,2]
        rn = np.tile(np.random.random(num_cases),(2,1)).T
        rn = (rn > prob_calc).astype(float)
        g = np.sum(rn, axis=1)
        cases[:,idx] = g
        current_sums += g * beta
    return cases

def main():
    # test efficient case generation
    """
    snp_props = generate_pss_model(num_phenos=2,num_snps=[10,0,0,10])
    cases1, cases2, controls = generate_case_matrix(snp_props,num_cases=20000, \
                                                        pi_prop=0.5,num_controls=10000, \
                                                        thresh=2.0,verbose=True)
    control_ave = np.mean(controls,axis=0)
    control_std = np.std(controls,axis=0)
    case1_ave = np.mean(cases1,axis=0)
    case1_std = np.std(cases1,axis=0)
    case2_ave = np.mean(cases2,axis=0)
    case2_std = np.std(cases2,axis=0)
    
    # old method
    cases1_s, cases2_s, controls_s = generate_cc_hl2(snp_props,num_cases=20000,pi_prop=0.50,num_controls=10000)    

    #controls_s = []
    #cases1_s = []
    #cases2_s = []
    #for i in range(10000):
    #    controls_s.append(Individual(snp_props,sample_posterior=True,case1=False,case2=False))
    #    cases1_s.append(Individual(snp_props,sample_posterior=True,case1=True,case2=False))
    #    cases2_s.append(Individual(snp_props,sample_posterior=True,case1=False,case2=True))
 
    controls_s_ave = []
    controls_s_std = []
    cases1_s_ave = []
    cases1_s_std = []
    cases2_s_ave = []
    cases2_s_std = []
    for i,snp in enumerate(range(len(snp_props))):
        controls_s_ave.append(np.mean([x.geno[snp] for x in controls_s]))
        controls_s_std.append(np.std([x.geno[snp] for x in controls_s]))
        cases1_s_ave.append(np.mean([x.geno[snp] for x in cases1_s]))
        cases1_s_std.append(np.std([x.geno[snp] for x in cases1_s]))
        cases2_s_ave.append(np.mean([x.geno[snp] for x in cases2_s]))
        cases2_s_std.append(np.std([x.geno[snp] for x in cases2_s]))
        
        print "control:", "{0:.2f}".format(control_ave[i]), \
                "control_s:", "{0:.2f}".format(controls_s_ave[i]), \
                "case1:", "{0:.2f}".format(case1_ave[i]), \
                "case1_s:", "{0:.2f}".format(cases1_s_ave[i]), \
                "case2:", "{0:.2f}".format(case2_ave[i]), \
                "case2_s:", "{0:.2f}".format(cases2_s_ave[i])
    
    xline = np.linspace(0,2,50)
    plt.plot(xline,xline,color='black')
    plt.scatter(control_ave, controls_s_ave, label="control inds", alpha=0.5)
    plt.scatter(case1_ave, cases1_s_ave, label="case1 inds", alpha=0.5)
    plt.scatter(case2_ave, cases2_s_ave, label="case2 inds", alpha=0.5)
    plt.xlabel("Vectorized case generation (N=10000)")
    plt.ylabel("Individual case generation (N=10000)")
    plt.legend()
    plt.show()
    sys.exit(0)
    """







    snp_props = generate_pss_model(num_phenos=2,num_snps=[15,0,0,15])
    num_inds = 20000

    # sampling method
    cases1_s, cases2_s, controls_s = generate_cc_hl2(snp_props,num_cases=num_inds*2,pi_prop=0.50,num_controls=10000)    

    # generative method    
    controls = []
    cases1 = []
    cases2 = []
    for i in range(num_inds):
        controls.append(Individual(snp_props,sample_posterior=True,case1=False,case2=False))
        cases1.append(Individual(snp_props,sample_posterior=True,case1=True,case2=False))
        cases2.append(Individual(snp_props,sample_posterior=True,case1=False,case2=True))

    controls_s_ave = []
    controls_s_std = []
    cases1_s_ave = []
    cases1_s_std = []
    cases2_s_ave = []
    cases2_s_std = []
    control_ave = []
    control_std = []
    case1_ave = []
    case1_std = []
    case2_ave = []
    case2_std = []
    for snp in range(len(snp_props)):
        controls_s_ave.append(np.mean([i.geno[snp] for i in controls_s]))
        controls_s_std.append(np.std([i.geno[snp] for i in controls_s]))
        cases1_s_ave.append(np.mean([i.geno[snp] for i in cases1_s]))
        cases1_s_std.append(np.std([i.geno[snp] for i in cases1_s]))
        cases2_s_ave.append(np.mean([i.geno[snp] for i in cases2_s]))
        cases2_s_std.append(np.std([i.geno[snp] for i in cases2_s]))

        control_ave.append(np.mean([i.geno[snp] for i in controls]))
        control_std.append(np.std([i.geno[snp] for i in controls]))
        case1_ave.append(np.mean([i.geno[snp] for i in cases1]))
        case1_std.append(np.std([i.geno[snp] for i in cases1]))
        case2_ave.append(np.mean([i.geno[snp] for i in cases2]))
        case2_std.append(np.std([i.geno[snp] for i in cases2]))

        

        print "control:", "{0:.2f}".format(control_ave[-1]), \
                "control_s:", "{0:.2f}".format(controls_s_ave[-1]), \
                "case1:", "{0:.2f}".format(case1_ave[-1]), \
                "case1_s:", "{0:.2f}".format(cases1_s_ave[-1]), \
                "case2:", "{0:.2f}".format(case2_ave[-1]), \
                "case2_s:", "{0:.2f}".format(cases2_s_ave[-1]), snp_props[snp].p, snp_props[snp].beta
    
    xline = np.linspace(0,2,50)
    plt.plot(xline,xline,color='black')
    plt.scatter(control_ave, controls_s_ave, label="control inds", alpha=0.5)
    plt.scatter(case1_ave, cases1_s_ave, label="case1 inds", alpha=0.5)
    plt.scatter(case2_ave, cases2_s_ave, label="case2 inds", alpha=0.5)
    plt.xlabel("Average genotype by direct generation (N=10000)")
    plt.ylabel("Average genotype by random sampling (N=10000)")
    plt.legend()
    plt.show()
    sys.exit(0)

if __name__=="__main__":
    #main()
    pass

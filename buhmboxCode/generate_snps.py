"""
SNP generator
Jie Yuan

This randomly generates a set of effect SNPs associated with a disease.
SNP properties are effect size and allele frequency. SNPs are assumed to
be independent.
"""
import math, random
import numpy as np
from pprint import pprint

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

class SNP():
    def __init__(self,name,p,beta):
        self.name=name
        self.p=p
        self.beta=beta
    def __str__(self):
        return "SNP %s p:%.4f beta:" % (self.name, self.p) + \
                ",".join(["%.4f"]*len(self.beta)) % tuple(self.beta)
                
def generate_pss_model_generalized(num_phenos=2, eff_size=0.1, eff_afreq=0.5,num_snps=np.array([[0, 50], [50, 0]])):
    snp_props = []

    p_low=eff_afreq
    p_high=eff_afreq
    b_loc = eff_size
    b_scale = 0.0 # no distribution: just take the mean every time  
    
    if num_snps.shape != tuple([2] * num_phenos):
        raise ValueError("Invalid num_snps passed to generate_pss_model_generalized")
    
    for index, val in np.ndenumerate(num_snps):
        name_counter = 0
        for i in range(int(val)):
            p = np.random.uniform(low=p_low,high=p_high)
            beta = [0]*num_phenos
            for i in range(len(index)):
                if index[i] == 1:
                    beta[i] = np.random.normal(loc=b_loc,scale=b_scale)
            snp_props.append(SNP("PSS-" + str(index) + "-" + str(name_counter), p, beta))
        name_counter += 1
    return snp_props

def generate_pss_model_generalized_vect(num_phenos=2, eff_size=0.1, eff_afreq=0.5,num_snps=np.array([[0, 50], [50, 0]])):
    all_ps = []
    all_betas = []

    p_low=eff_afreq
    p_high=eff_afreq
    b_loc = eff_size
    b_scale = 0.0 # no distribution: just take the mean every time  
    
    if num_snps.shape != tuple([2] * num_phenos):
        raise ValueError("Invalid num_snps passed to generate_pss_model_generalized")
    
    for index, val in np.ndenumerate(num_snps):
        name_counter = 0
        n = int(val)
        p = np.random.uniform(low=p_low,high=p_high, size=n)
        beta = np.random.normal(loc=b_loc,scale=b_scale, size=(n, num_phenos)) * np.array(index).reshape(1, num_phenos)
        all_ps.append(p)
        all_betas.append(beta)
    return np.hstack(all_ps), np.vstack(all_betas)

def generate_pss_model(num_phenos=2, nsnps=50, eff_size=0.1, eff_afreq=0.5,num_snps=None):
    snp_props = []

    p_low=eff_afreq
    p_high=eff_afreq
    b_loc = eff_size
    b_scale = 0.0 # no distribution: just take the mean every time  
    if num_snps is None:
        num_snps = [nsnps/2, 0, 0, nsnps/2]
    
    name_counter = 0
    # snps affecting only trait1
    for i in range(num_snps[0]):
        p = np.random.uniform(low=p_low,high=p_high)
        beta = [0]*num_phenos
        beta[0] = np.random.normal(loc=b_loc,scale=b_scale)
        # if beta[0] < 0:
        #     beta[0] *= -1
        snp_props.append(SNP("PSS1-"+str(name_counter),p,beta))
        name_counter += 1

    # snps affecting both trait1 and trait2
    name_counter = 0
    for i in range(num_snps[1]):
        p = np.random.uniform(low=p_low,high=p_high)
        beta = [0]*num_phenos
        beta[0] = np.random.normal(loc=b_loc,scale=b_scale)
        if beta[0] < 0:
            beta[0] *= -1
        beta[1] = np.random.normal(loc=b_loc,scale=b_scale)
        if beta[1] < 0:
            beta[1] *= -1
        snp_props.append(SNP("BOTH-"+str(name_counter),p,beta))
        name_counter += 1

    # snps affecting neither trait1 nor trait2
    name_counter = 0
    for i in range(num_snps[2]):
        p = np.random.uniform(low=p_low,high=p_high)
        beta = [0]*num_phenos
        snp_props.append(SNP("NEITHER-"+str(name_counter),p,beta))
        name_counter += 1
    
    # snps affecting only trait2
    name_counter = 0
    for i in range(num_snps[3]):
        p = np.random.uniform(low=p_low,high=p_high)
        beta = [0]*num_phenos
        beta[1] = np.random.normal(loc=b_loc,scale=b_scale)
        if beta[1] < 0:
            beta[1] *= -1
        snp_props.append(SNP("PSS2-"+str(name_counter),p,beta))
        name_counter += 1

    return snp_props

def calc_heritability(snp_props):
    t0_h = 0.0
    t1_h = 0.0
    for snp in snp_props:
        t0_h += snp.beta[0]**2 * 2 * snp.p * (1 - snp.p)
        t1_h += snp.beta[1]**2 * 2 * snp.p * (1 - snp.p)
    return t0_h, t1_h
    
def plot_snp_props(snp_props):
    pvalues = [x.p for x in snp_props]
    t0_betas = [x.beta[0] for x in snp_props]
    t1_betas = [x.beta[1] for x in snp_props]
    plt.scatter(pvalues,t0_betas,color='b',label='PSS1')
    plt.scatter(pvalues,t1_betas,color='r',label='PSS2')
    plt.legend()
    plt.xlim(xmin=0,xmax=1.0)
    plt.ylim(ymin=0)
    plt.xlabel('Effect Allele Frequency')
    plt.ylabel('Beta')
    plt.show()

def print_snp_props(snp_props):
    mystr = "[\n"
    for snp in snp_props:
        mystr += "  " + str(snp) + "\n"
    mystr += "]"
    print mystr

def main():
    snp_props = generate_pss_model(num_phenos=2)
    t1,t2 = calc_heritability(snp_props)
    print t1,t2
    plot_snp_props(snp_props)
    
if __name__=="__main__":
    main()

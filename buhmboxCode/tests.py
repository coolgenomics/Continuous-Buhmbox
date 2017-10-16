from generate_snps import *
from generate_population import *
from get_max_cluster_singlerun_Rcorrection_Jennerich_full_matrix import *
import numpy as np

def main1():
    num_snps = 100
    num_inds = 10000
    independent_snps = generate_pss_model_generalized(num_phenos=2, num_snps=np.array([[0, num_snps], [num_snps, 0]]))
    pleiotropy_snps = generate_pss_model_generalized(num_phenos=2, num_snps=np.array([[0, num_snps/2], [num_snps/2, num_snps/2]]))
    b = np.zeros(8).reshape((2, 2, 2))
    b[1, 0, 0] = num_snps/4
    b[0, 0, 1] = num_snps/4
    b[1, 0, 1] = num_snps*3/4
    b[0, 1, 0] = num_snps
    hetero_snps = generate_pss_model_generalized(num_phenos=3, num_snps=b)
    
    independent_pop = generate_population(independent_snps, num_inds=num_inds)
    pleiotropic_pop = generate_population(pleiotropy_snps, num_inds=num_inds)
    hetero_pop = generate_population_hetero(hetero_snps, num_inds=num_inds)
    
    plot_inds(independent_pop)
    plot_inds(pleiotropic_pop)
    plot_inds(hetero_pop)
    
    plot_inds_alts(hetero_pop)    

def get_mats_from_pop(pop, phen, z_thresh):
    phenos_rows = []
    genos_rows = []
    for indiv in pop:
        phenos_rows.append(indiv.pheno)
        genos_rows.append(indiv.geno)
    genos = np.vstack(genos_rows)
    phenos = np.vstack(phenos_rows)
    mu = np.mean(phenos[:, phen])
    sigma = np.std(phenos[:, phen])
    
    cases_indices = np.where(phenos[:, phen] > (mu + z_thresh * sigma))
    controls_indices = np.where(phenos[:, phen] <= (mu + z_thresh * sigma))
    cases = genos[cases_indices]
    controls = genos[controls_indices]
    return cases, controls
    
def get_clist(snps, phens):
    snp_betas_rows = [snp.beta for snp in snps]
    snp_betas = np.vstack(snp_betas_rows)
    return tuple(set(np.where(snp_betas[:, phens] > 0)[0]))

def run_buhmbox_on_pop(pop, snps, snp_phens=0, case_phen=1, z=1.5):
    cases, controls = get_mats_from_pop(pop, case_phen, z)
    num_cases, _ = cases.shape
    controls_sub = controls[:num_cases, :]
    clist = get_clist(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return buhmbox(cases, controls_sub, clist, snps)

def main2():
    num_snps = 100
    num_inds = 100000
    independent_snps = generate_pss_model_generalized(num_phenos=2, num_snps=np.array([[0, num_snps], [num_snps, 0]]))
    pleiotropy_snps = generate_pss_model_generalized(num_phenos=2, num_snps=np.array([[0, num_snps/2], [num_snps/2, num_snps/2]]))
    b = np.zeros(8).reshape((2, 2, 2))
    b[1, 0, 0] = num_snps/4
    b[0, 0, 1] = num_snps/4
    b[1, 0, 1] = num_snps*3/4
    b[0, 1, 0] = num_snps
    hetero_snps = generate_pss_model_generalized(num_phenos=3, num_snps=b)
    
    independent_pop = generate_population(independent_snps, num_inds=num_inds)
    pleiotropic_pop = generate_population(pleiotropy_snps, num_inds=num_inds)
    hetero_pop = generate_population_hetero(hetero_snps, num_inds=num_inds)
    
    
    plot_inds(independent_pop)
    plot_inds(pleiotropic_pop)
    plot_inds(hetero_pop)
    
    plot_inds_alts(hetero_pop)    
    print run_buhmbox_on_pop(independent_pop, independent_snps)
    print run_buhmbox_on_pop(pleiotropic_pop, pleiotropy_snps)
    print run_buhmbox_on_pop(hetero_pop, hetero_snps)
    

if __name__=="__main__":
    main2()

from generate_snps import *
from generate_population import *
from get_max_cluster_singlerun_Rcorrection_Jennerich_full_matrix import *
import numpy as np
import pickle
import time

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

def get_mats_from_pop_vect(pop, phen, z_thresh):
    genos, phenos, _ = pop
    mu = np.mean(phenos[:, phen])
    sigma = np.std(phenos[:, phen])
    
    cases_indices = np.where(phenos[:, phen] > (mu + z_thresh * sigma))
    controls_indices = np.where(phenos[:, phen] <= (mu + z_thresh * sigma))
    cases = genos[cases_indices]
    controls = genos[controls_indices]
    return cases, controls
    
def get_clist_vect(snps, phens):
    _, snp_betas = snps
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

def run_vect_buhmbox_on_pop(pop, snps, snp_phens=0, case_phen=1, z=1.5):
    cases, controls = get_mats_from_pop(pop, case_phen, z)
    num_cases, _ = cases.shape
    controls_sub = controls[:num_cases, :]
    clist = get_clist(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return buhmbox_vectorized(cases, controls_sub, clist, snps)

def run_cont_buhmbox_on_pop(pop, snps, snp_phens=0, case_phen=1):
    phenos_rows = []
    genos_rows = []
    for indiv in pop:
        phenos_rows.append(indiv.pheno)
        genos_rows.append(indiv.geno)
    genos = np.vstack(genos_rows)
    phenos = np.vstack(phenos_rows)
    clist = get_clist(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return continuous_buhmbox(genos, phenos[:, case_phen], clist, snps)

def run_vect_cont_buhmbox_on_pop(pop, snps, snp_phens=0, case_phen=1):
    phenos_rows = []
    genos_rows = []
    for indiv in pop:
        phenos_rows.append(indiv.pheno)
        genos_rows.append(indiv.geno)
    genos = np.vstack(genos_rows)
    phenos = np.vstack(phenos_rows)
    clist = get_clist(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return continuous_buhmbox_vect(genos, phenos[:, case_phen], clist, snps)

def run_buhmbox_on_vect_pop(pop, snps, snp_phens=0, case_phen=1, z=1.5):
    cases, controls = get_mats_from_pop_vect(pop, case_phen, z)
    num_cases, _ = cases.shape
    controls_sub = controls[:num_cases, :]
    clist = get_clist_vect(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return buhmbox(cases, controls_sub, clist, snps)

def run_vect_buhmbox_on_vect_pop(pop, snps, snp_phens=0, case_phen=1, z=1.5):
    cases, controls = get_mats_from_pop_vect(pop, case_phen, z)
    num_cases, _ = cases.shape
    controls_sub = controls[:num_cases, :]
    clist = get_clist_vect(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return buhmbox_vectorized(cases, controls_sub, clist, snps)

def run_cont_buhmbox_on_vect_pop(pop, snps, snp_phens=0, case_phen=1):
    genos, phenos, _ = pop
    clist = get_clist_vect(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return continuous_buhmbox(genos, phenos[:, case_phen], clist, snps)

def run_vect_cont_buhmbox_on_vect_pop(pop, snps, snp_phens=0, case_phen=1):
    genos, phenos, _ = pop
    clist = get_clist_vect(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return continuous_buhmbox_vect(genos, phenos[:, case_phen], clist, snps)

def main2():
    runs = 1
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
    
    ib = []
    pb = []
    hb = []
    ic = []
    pc = []
    hc = []
    start = time.time()
    for i in range(0, runs):    
        independent_pop = generate_population(independent_snps, num_inds=num_inds)
        pleiotropic_pop = generate_population(pleiotropy_snps, num_inds=num_inds)
        hetero_pop = generate_population_hetero(hetero_snps, num_inds=num_inds)

        '''
        plot_inds(independent_pop)
        plot_inds(pleiotropic_pop)
        plot_inds(hetero_pop)
        plot_inds_alts(hetero_pop)
        '''

        ib.append(run_buhmbox_on_pop(independent_pop, independent_snps))
        pb.append(run_buhmbox_on_pop(pleiotropic_pop, pleiotropy_snps))
        hb.append(run_buhmbox_on_pop(hetero_pop, hetero_snps))
        ic.append(run_cont_buhmbox_on_pop(independent_pop, independent_snps))
        pc.append(run_cont_buhmbox_on_pop(pleiotropic_pop, pleiotropy_snps))
        hc.append(run_cont_buhmbox_on_pop(hetero_pop, hetero_snps))
    arrs = [ib, pb, hb, ic, pc, hc]
    means = [np.mean(arr) for arr in arrs]
    stds = [np.std(arr) for arr in arrs]
    info = (arrs, means, stds)
    with open("info.pickle", "wb") as f:
        pickle.dump(info, f)
    print(time.time()-start)

def main3():
    runs = 50
    num_snps = 100
    num_inds = 10000
    independent_snps = generate_pss_model_generalized_vect(num_phenos=2, num_snps=np.array([[0, num_snps], [num_snps, 0]]))
    pleiotropy_snps = generate_pss_model_generalized_vect(num_phenos=2, num_snps=np.array([[0, num_snps/2], [num_snps/2, num_snps/2]]))
    b = np.zeros(8).reshape((2, 2, 2))
    b[1, 0, 0] = num_snps/4
    b[0, 0, 1] = num_snps/4
    b[1, 0, 1] = num_snps*3/4
    b[0, 1, 0] = num_snps
    hetero_snps = generate_pss_model_generalized_vect(num_phenos=3, num_snps=b)
    
    ib = []
    pb = []
    hb = []
    ic = []
    pc = []
    hc = []
    start = time.time()
    for i in range(0, runs):    
        independent_pop = generate_population_vect(independent_snps, num_inds=num_inds)
        pleiotropic_pop = generate_population_vect(pleiotropy_snps, num_inds=num_inds)
        hetero_pop = generate_hetero_population_vect(hetero_snps, num_inds=num_inds)

        '''
        plot_inds(independent_pop)
        plot_inds(pleiotropic_pop)
        plot_inds(hetero_pop)
        plot_inds_alts(hetero_pop)
        '''

        ib.append(run_vect_buhmbox_on_vect_pop(independent_pop, independent_snps))
        pb.append(run_vect_buhmbox_on_vect_pop(pleiotropic_pop, pleiotropy_snps))
        hb.append(run_vect_buhmbox_on_vect_pop(hetero_pop, hetero_snps))
        ic.append(run_vect_cont_buhmbox_on_vect_pop(independent_pop, independent_snps))
        pc.append(run_vect_cont_buhmbox_on_vect_pop(pleiotropic_pop, pleiotropy_snps))
        hc.append(run_vect_cont_buhmbox_on_vect_pop(hetero_pop, hetero_snps))
    arrs = [ib, pb, hb, ic, pc, hc]
    means = [np.mean(arr) for arr in arrs]
    stds = [np.std(arr) for arr in arrs]
    info = (arrs, means, stds)
    with open("info.pickle", "wb") as f:
        pickle.dump(info, f)
    print(time.time()-start)

def print_info():
    with open("info.pickle", "rb") as f:
        arrs, means, stds = pickle.load(f)
    for arr in arrs:
        print(arr)
    for mean in means:
        print(mean)
    for std in stds:
        print(std)

def main4():
    """
    runs = 1
    num_snps = 100
    num_inds = 100000
    start = time.time()
    independent_snps = generate_pss_model_generalized(num_phenos=2, num_snps=np.array([[0, num_snps], [num_snps, 0]]))
    pleiotropy_snps = generate_pss_model_generalized(num_phenos=2, num_snps=np.array([[0, num_snps/2], [num_snps/2, num_snps/2]]))
    b = np.zeros(8).reshape((2, 2, 2))
    b[1, 0, 0] = num_snps/4
    b[0, 0, 1] = num_snps/4
    b[1, 0, 1] = num_snps*3/4
    b[0, 1, 0] = num_snps
    hetero_snps = generate_pss_model_generalized(num_phenos=3, num_snps=b)
    print time.time() - start
    start = time.time()
    independent_pop = generate_population(independent_snps, num_inds=num_inds)
    pleiotropic_pop = generate_population(pleiotropy_snps, num_inds=num_inds)
    hetero_pop = generate_population_hetero(hetero_snps, num_inds=num_inds)
    print time.time() - start
    isn, psn, hsn, ip, pp, hp = independent_snps, pleiotropy_snps, hetero_snps, independent_pop, pleiotropic_pop, hetero_pop
    for sn, p in ((isn, ip), (psn, pp), (hsn, hp)):
        start1 = time.time()
        val1 = run_buhmbox_on_pop(p, sn)
        print(time.time()-start1, val1)
        start2 = time.time()
        val2 = run_vect_buhmbox_on_pop(p, sn)
        print(time.time()-start2, val2)
        print

    for sn, p in ((isn, ip), (psn, pp), (hsn, hp)):
        start1 = time.time()
        val1 = run_cont_buhmbox_on_pop(p, sn)
        print(time.time()-start1, val1)
        start2 = time.time()
        val2 = run_vect_cont_buhmbox_on_pop(p, sn)
        print(time.time()-start2, val2)
        print
    """
        
    runs = 1
    num_snps = 100
    num_inds = 100000
    start = time.time()
    independent_snps = generate_pss_model_generalized_vect(num_phenos=2, num_snps=np.array([[0, num_snps], [num_snps, 0]]))
    pleiotropy_snps = generate_pss_model_generalized_vect(num_phenos=2, num_snps=np.array([[0, num_snps/2], [num_snps/2, num_snps/2]]))
    b = np.zeros(8).reshape((2, 2, 2))
    b[1, 0, 0] = num_snps/4
    b[0, 0, 1] = num_snps/4
    b[1, 0, 1] = num_snps*3/4
    b[0, 1, 0] = num_snps
    hetero_snps = generate_pss_model_generalized_vect(num_phenos=3, num_snps=b)
    print time.time() - start
    start = time.time()
    independent_pop = generate_population_vect(independent_snps, num_inds=num_inds)
    pleiotropic_pop = generate_population_vect(pleiotropy_snps, num_inds=num_inds)
    hetero_pop = generate_hetero_population_vect(hetero_snps, num_inds=num_inds)
    print time.time() - start
    isn, psn, hsn, ip, pp, hp = independent_snps, pleiotropy_snps, hetero_snps, independent_pop, pleiotropic_pop, hetero_pop
    for sn, p in ((isn, ip), (psn, pp), (hsn, hp)):
        start1 = time.time()
        val1 = run_buhmbox_on_vect_pop(p, sn)
        print(time.time()-start1, val1)
        start2 = time.time()
        val2 = run_vect_buhmbox_on_vect_pop(p, sn)
        print(time.time()-start2, val2)
        print

    for sn, p in ((isn, ip), (psn, pp), (hsn, hp)):
        start1 = time.time()
        val1 = run_cont_buhmbox_on_vect_pop(p, sn)
        print(time.time()-start1, val1)
        start2 = time.time()
        val2 = run_vect_cont_buhmbox_on_vect_pop(p, sn)
        print(time.time()-start2, val2)
        print


if __name__=="__main__":
    #main4()
    main3()
    print_info()
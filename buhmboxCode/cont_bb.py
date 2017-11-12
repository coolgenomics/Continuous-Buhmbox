import os.path
import numpy as np
import time
from scipy.stats import rankdata
import pickle

FILE_PATH = "info.pickle"
TEST_FILE_PATH = "info-test.pickle"

def generate_pss_model_generalized(num_phenos=2, eff_size=0.1, eff_afreq=0.5,num_snps=np.array([[0, 50], [50, 0]])):
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

def generate_hetero_population(snps, num_inds=10000):
    """
    snps=[snp_ps, snp_betas]
    snp_ps: numpy length num_snps array with rafs
    snp_betas: numpy (num_snps, num phenos) matrix with betas
    """
    snp_ps, snp_betas = snps
    assert len(snp_ps) == len(snp_betas)
    num_snps = len(snp_ps)
    assert num_snps > 0
    num_phenos = len(snp_betas[0])

    # sample SNPs according to SNP props
    randoms = np.random.rand(num_inds, num_snps, 1)
    snp_ps_all = np.repeat(snp_ps.reshape(1, num_snps, 1), num_inds, axis=0)
    geno = (randoms < snp_ps_all**2.0).astype(float) + (randoms < snp_ps_all**2.0 + 2.0*snp_ps_all*(1.0-snp_ps_all)).astype(float)
    assert geno.shape == (num_inds, num_snps, 1)
    betas_all = np.repeat(snp_betas.reshape(1, num_snps, num_phenos), num_inds, axis=0)
    pheno = np.sum(np.repeat(geno, num_phenos, axis=2) * betas_all, axis=1)
    assert pheno.shape == (num_inds, num_phenos)
    
    alts = pheno[:, -1] > pheno[:, -2]
    alts_float = alts.astype(float)
    new_pheno = (pheno[:, -1] * alts_float) + (pheno[:, -2] * (1 - alts_float))
    real_phenos = np.hstack((pheno[:, :-2], new_pheno.reshape(num_inds, 1)))
    return geno.reshape(num_inds, num_snps), real_phenos, alts

def generate_population(snps, num_inds=10000):
    """
    snps=[snp_ps, snp_betas]
    snp_ps: numpy length num_snps array with rafs
    snp_betas: numpy (num_snps, num phenos) matrix with betas
    """
    snp_ps, snp_betas = snps
    assert len(snp_ps) == len(snp_betas)
    num_snps = len(snp_ps)
    assert num_snps > 0
    num_phenos = len(snp_betas[0])

    # sample SNPs according to SNP props
    randoms = np.random.rand(num_inds, num_snps, 1)
    snp_ps_all = np.repeat(snp_ps.reshape(1, num_snps, 1), num_inds, axis=0)
    geno = (randoms < snp_ps_all**2.0).astype(float) + (randoms < snp_ps_all**2.0 + 2.0*snp_ps_all*(1.0-snp_ps_all)).astype(float)
    assert geno.shape == (num_inds, num_snps, 1)
    betas_all = np.repeat(snp_betas.reshape(1, num_snps, num_phenos), num_inds, axis=0)
    pheno = np.sum(np.repeat(geno, num_phenos, axis=2) * betas_all, axis=1)
    assert pheno.shape == (num_inds, num_phenos)
    
    return geno.reshape(num_inds, num_snps), pheno, np.zeros(num_inds)

def buhmbox(cases,controls,clist,snp_props):
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

def get_mats_from_pop(pop, phen, z_thresh):
    genos, phenos, _ = pop
    mu = np.mean(phenos[:, phen])
    sigma = np.std(phenos[:, phen])
    
    cases_indices = np.where(phenos[:, phen] > (mu + z_thresh * sigma))
    controls_indices = np.where(phenos[:, phen] <= (mu + z_thresh * sigma))
    cases = genos[cases_indices]
    controls = genos[controls_indices]
    return cases, controls
    
def get_clist(snps, phens):
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

def run_cont_buhmbox_on_pop(pop, snps, snp_phens=0, case_phen=1):
    genos, phenos, _ = pop
    clist = get_clist(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return continuous_buhmbox(genos, phenos[:, case_phen], clist, snps)

def main3(file_path, runs=1, num_snps=100, num_inds=100000):
    start = time.time()
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
    for i in range(0, runs):    
        independent_pop = generate_population(independent_snps, num_inds=num_inds)
        pleiotropic_pop = generate_population(pleiotropy_snps, num_inds=num_inds)
        hetero_pop = generate_hetero_population(hetero_snps, num_inds=num_inds)

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
    with open(file_path, "wb") as f:
        pickle.dump(info, f)
    print(time.time()-start)

def print_info(file_path):
    with open(file_path, "rb") as f:
        arrs, means, stds = pickle.load(f)
    for arr in arrs:
        print(arr)
    for mean in means:
        print(mean)
    for std in stds:
        print(std)

if __name__=="__main__":
    main3(TEST_FILE_PATH, runs=1, num_snps=100, num_inds=100000)
    print_info(TEST_FILE_PATH)
    """
    if not os.path.exists(FILE_PATH):
        main3(FILE_PATH, runs=1, num_snps=100, num_inds=100000)
    print_info(FILE_PATH)
    """

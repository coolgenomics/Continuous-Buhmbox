from generate_snps import *
from generate_population import *
from get_max_cluster_singlerun_Rcorrection_Jennerich_full_matrix import *

def main():
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
    
    #plot_inds(independent_pop)
    #plot_inds(pleiotropic_pop)
    #plot_inds(hetero_pop)
    
    #plot_inds_alts(hetero_pop)    
    

if __name__=="__main__":
    main()

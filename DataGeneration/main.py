from DataGeneration import DataGen

if __name__ == '__main__':
    number_of_dataset = 500
    num_envs = 8
    num_data_each_env = 6000
    num_spurious_features = 4
    num_informative_features = 4
    num_redundant_features = 0
    unseen_domains = [5, 6, 7, 8]
    low_mean = -500
    high_mean = 500
    low_cov = 1
    high_cov = 100
    DataGen.generate_multi_datasets(number_of_dataset, num_envs, num_data_each_env, num_informative_features,
                            num_spurious_features, num_redundant_features, unseen_domains, low_mean, high_mean, low_cov, high_cov)
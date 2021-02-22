from collections import Counter

recording_length = 952.2185
firing_rate = 4.02008
H_spiking = 0.09843

embedding_past_range_set = [0.005, 0.00998, 0.15811, 1.25594, 5.0]
embedding_number_of_bins_set = [1, 3, 5]
embedding_scaling_exponent_set = {'number_of_scalings': 10,
                                  'min_first_bin_size': 0.005,
                                  'min_step_for_scaling': 0.01}
estimation_method = 'all'

T_D_bbc = 1.25594
R_tot_bbc = 0.11185
AIS_tot_bbc = 0.01101
opt_number_of_bins_d_bbc = 5.00000
opt_scaling_k_bbc = 0.44001
opt_first_bin_size_bbc = 0.01399
bbc_term = 0.001737059851451546

T_D_shuffling = 1.25594
R_tot_shuffling = 0.11087
AIS_tot_shuffling = 0.01091
opt_number_of_bins_d_shuffling = 5.00000
opt_scaling_k_shuffling = 0.44001
opt_first_bin_size_shuffling = 0.01399

num_embeddings = 69
embedding = (1.25594, 5, 0.4400051332162487)
symbol_counts = Counter({8: 45658, 16: 25801, 40: 23527, 32: 19786,
                         0: 19226, 4: 13280, 36: 8173, 24: 6434,
                         48: 5306, 20: 5138, 2: 3963, 34: 2725,
                         18: 2265, 12: 1388, 1: 1304, 10: 955,
                         33: 939, 17: 829, 44: 749, 52: 635,
                         56: 514, 42: 459, 9: 426, 50: 288, 41: 196,
                         49: 104, 28: 50, 26: 32, 25: 16, 58: 6,
                         6: 5, 60: 5, 5: 3, 37: 3, 57: 2, 38: 1})
past_symbol_counts = Counter({4: 46084, 8: 26630, 20: 23723,
                              16: 20725, 0: 20530, 2: 13283, 18: 8176,
                              12: 6450, 24: 5410, 10: 5138, 1: 3963,
                              17: 2725, 9: 2265, 6: 1388, 5: 955,
                              22: 749, 26: 635, 28: 516, 21: 459,
                              25: 288, 14: 50, 13: 32, 29: 6, 3: 5,
                              30: 5, 19: 1})

max_Rs_bbc = {0.005: 0.004159491380826222,
              0.00998: 0.008403613970019453,
              0.15811: 0.10777398676863034,
              1.25594: 0.11185019362783793,
              5.0: 0.10888327025866504}

max_Rs_shuffling = {0.005: 0.004187714772907364,
                    0.00998: 0.008431566628498323,
                    0.15811: 0.1069914609059317,
                    1.25594: 0.1108672485231234,
                    5.0: 0.10848116994653662}

CI_lo = 0.107
CI_hi = 0.116

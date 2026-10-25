## u1000

data: {"events": 4768010, "users": 1000, "items": 279380, "split": {"train": [266, 286], "val": [287, 293], "test": [294, 300]}, "rows": {"train": 211850, "val": 74606, "test": 71659}, "pos_rate": {"train": 0.6658531980174652, "val": 0.6677210948181111, "test": 0.6861803820873861}, "share_rec_listens": 0.4852954486892164}

timings (s): {"load_s": 14.6, "engine_build_s": 7.6, "session_s": 1.2, "session_grammar_s": 6.8, "manual_features_s": 4.1, "autofe_search_s": 257.8, "autofe_stage1_s": 203.4, "autofe_stage2_s": 54.4, "autofe_n_candidates": 2746, "autofe_n_pool": 251, "autofe_best_n": 45, "autofe_full_search_s": 252.9, "autofe_full_stage1_s": 201.3, "autofe_full_stage2_s": 51.7, "autofe_full_n_candidates": 2832, "autofe_full_n_pool": 282, "autofe_full_best_n": 100, "auto_features_s": 5.5, "autofe_full_features_s": 13.0, "peak_rss_gb": 5.1}

| model                       |   n_feat |   val_logloss |   val_ndcg@10 |   test_logloss |   test_auc |   test_gauc |   test_ndcg@10 |    fit_s |
|:----------------------------|---------:|--------------:|--------------:|---------------:|-----------:|------------:|---------------:|---------:|
| pop_item_fullrate           |        1 |        0.9831 |        0.7142 |         0.9478 |     0.6372 |      0.5368 |         0.7209 | nan      |
| user_fullrate               |        1 |        0.4822 |        0.6982 |         0.4708 |     0.8228 |      0.5000 |         0.7130 | nan      |
| ctx_only                    |        3 |        0.6295 |        0.7198 |         0.6167 |     0.5688 |      0.5333 |         0.7281 |   1.5239 |
| manual                      |       58 |        0.4550 |        0.7531 |         0.4507 |     0.8379 |      0.5761 |         0.7572 |   4.3802 |
| autofe                      |       48 |        0.4554 |        0.7583 |         0.4478 |     0.8409 |      0.5714 |         0.7556 |   3.7695 |
| manual+sess                 |       66 |        0.3487 |        0.7748 |         0.3469 |     0.9101 |      0.6592 |         0.7719 |   5.3669 |
| random_sameN+sess           |       66 |        0.3477 |        0.7770 |         0.3463 |     0.9104 |      0.6570 |         0.7712 |   5.7305 |
| manual_retuned_windows+sess |       48 |        0.3486 |        0.7752 |         0.3460 |     0.9106 |      0.6613 |         0.7750 |   5.5598 |
| autofe+sess                 |       56 |        0.3477 |        0.7761 |         0.3458 |     0.9108 |      0.6616 |         0.7744 |   5.2523 |
| autofe_sameN+sess           |       66 |        0.3483 |        0.7750 |         0.3458 |     0.9107 |      0.6595 |         0.7755 |   5.9008 |
| manual∪autofe+sess          |      110 |        0.3477 |        0.7765 |         0.3455 |     0.9109 |      0.6633 |         0.7760 |   7.6814 |
| autofe_full                 |      103 |        0.3538 |        0.7559 |         0.3535 |     0.9063 |      0.6511 |         0.7589 |   7.4306 |
| autofe_full+sess            |      111 |        0.3431 |        0.7733 |         0.3412 |     0.9134 |      0.6716 |         0.7789 |   7.7672 |
| manual+sess_lambdarank      |       66 |        0.8069 |        0.8344 |         0.8192 |     0.6512 |      0.7016 |         0.8287 |  12.2551 |
| autofe+sess_lambdarank      |       56 |        0.8354 |        0.8326 |         0.8478 |     0.6415 |      0.7050 |         0.8312 |  13.5415 |
| autofe_full_lambdarank      |      103 |        0.6107 |        0.8009 |         0.6155 |     0.8004 |      0.6651 |         0.7938 |  10.9610 |

Paired bootstrap по пользователям, разница с manual+sess (test):

| model                       | gauc                       | logloss                    | ndcg@10                    |
|:----------------------------|:---------------------------|:---------------------------|:---------------------------|
| autofe                      | -0.0878 [-0.1049, -0.0699] | +0.1009 [+0.0865, +0.1163] | -0.0163 [-0.0276, -0.0042] |
| autofe+sess                 | +0.0024 [-0.0021, +0.0067] | -0.0012 [-0.0022, -0.0002] | +0.0025 [-0.0014, +0.0062] |
| autofe+sess_lambdarank      | +0.0457 [+0.0339, +0.0588] | +0.5009 [+0.4465, +0.5547] | +0.0593 [+0.0519, +0.0663] |
| autofe_full                 | -0.0082 [-0.0155, -0.0012] | +0.0066 [+0.0041, +0.0091] | -0.0130 [-0.0191, -0.0062] |
| autofe_full+sess            | +0.0123 [+0.0072, +0.0178] | -0.0057 [-0.0072, -0.0044] | +0.0071 [+0.0024, +0.0124] |
| autofe_full_lambdarank      | +0.0059 [-0.0028, +0.0159] | +0.2686 [+0.2418, +0.2936] | +0.0219 [+0.0137, +0.0305] |
| autofe_sameN+sess           | +0.0003 [-0.0040, +0.0045] | -0.0012 [-0.0022, -0.0002] | +0.0036 [-0.0004, +0.0080] |
| ctx_only                    | -0.1259 [-0.1466, -0.1056] | +0.2698 [+0.2481, +0.2939] | -0.0437 [-0.0568, -0.0304] |
| manual                      | -0.0831 [-0.1018, -0.0637] | +0.1038 [+0.0890, +0.1200] | -0.0147 [-0.0261, -0.0028] |
| manual+sess                 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| manual+sess_lambdarank      | +0.0423 [+0.0307, +0.0549] | +0.4722 [+0.4221, +0.5215] | +0.0568 [+0.0501, +0.0637] |
| manual_retuned_windows+sess | +0.0020 [-0.0016, +0.0059] | -0.0009 [-0.0019, +0.0000] | +0.0031 [-0.0001, +0.0066] |
| manual∪autofe+sess          | +0.0041 [+0.0003, +0.0084] | -0.0014 [-0.0022, -0.0007] | +0.0042 [+0.0004, +0.0076] |
| random_sameN+sess           | -0.0022 [-0.0074, +0.0026] | -0.0006 [-0.0016, +0.0003] | -0.0006 [-0.0046, +0.0033] |

Лучший масштаб времени по семействам (топ-25 по gain):

| family                                     | feature                                          |    gain |
|:-------------------------------------------|:-------------------------------------------------|--------:|
| rate__user__full_rec_per_listen_rec        | drate__user__full_rec_per_listen_rec__h30        | 0.16817 |
| rate__user__full_per_listen                | drate__user__full_per_listen__h14                | 0.15739 |
| rate__user__ratio_sum_per_listen           | drate__user__ratio_sum_per_listen__h14           | 0.15665 |
| rate__user__early_skip_per_listen          | drate__user__early_skip_per_listen__h14          | 0.15254 |
| cnt__user__early_skip                      | dcnt__user__early_skip__h7                       | 0.10714 |
| rate__user_artist__ratio_sum_per_listen    | drate__user_artist__ratio_sum_per_listen__h90    | 0.09604 |
| rate__user_artist__full_rec_per_listen_rec | drate__user_artist__full_rec_per_listen_rec__h90 | 0.09547 |
| rate__user_artist__full_per_listen         | drate__user_artist__full_per_listen__h90         | 0.09317 |
| rate__user_artist__early_skip_per_listen   | drate__user_artist__early_skip_per_listen__h90   | 0.08437 |
| rate__user_album__ratio_sum_per_listen     | drate__user_album__ratio_sum_per_listen__h90     | 0.06479 |
| rate__user_album__full_per_listen          | drate__user_album__full_per_listen__h90          | 0.06097 |
| rate__user_album__full_rec_per_listen_rec  | drate__user_album__full_rec_per_listen_rec__h90  | 0.06011 |
| rate__user_album__early_skip_per_listen    | drate__user_album__early_skip_per_listen__h90    | 0.05519 |
| rate__user_item__ratio_sum_per_listen      | drate__user_item__ratio_sum_per_listen__h90      | 0.05217 |
| cnt__user__full_rec                        | dcnt__user__full_rec__h7                         | 0.05143 |
| rate__user_item__full_per_listen           | drate__user_item__full_per_listen__h90           | 0.04803 |
| rate__user_item__full_rec_per_listen_rec   | drate__user_item__full_rec_per_listen_rec__h90   | 0.04721 |
| rate__user_item__early_skip_per_listen     | drate__user_item__early_skip_per_listen__h90     | 0.04283 |
| cnt__user_artist__ratio_sum                | cnt__user_artist__ratio_sum__w21                 | 0.03414 |
| cnt__user_album__ratio_sum                 | cnt__user_album__ratio_sum__w90                  | 0.03064 |
| rate__album__full_rec_per_listen_rec       | drate__album__full_rec_per_listen_rec__h90       | 0.02870 |
| rate__artist__full_rec_per_listen_rec      | drate__artist__full_rec_per_listen_rec__h30      | 0.02837 |
| cnt__user__full                            | cnt__user__full__w30                             | 0.02771 |
| rate__album__ratio_sum_per_listen          | drate__album__ratio_sum_per_listen__h90          | 0.02613 |
| rate__item__full_rec_per_listen_rec        | drate__item__full_rec_per_listen_rec__h90        | 0.02546 |

Распределение лучших масштабов: {"all/other": 45, "h=90.0": 23, "h=30.0": 13, "h=14.0": 10, "h=365.0": 7, "h=180.0": 7, "h=3.0": 6, "w=60.0": 6, "w=1.0": 6, "w=180.0": 5, "h=7.0": 5, "w=7.0": 4, "w=90.0": 4, "w=21.0": 3, "h=1.0": 3, "w=30.0": 3, "w=3.0": 2, "w=2.0": 2, "w=120.0": 1, "w=45.0": 1}

Топ-15 признаков по gain (manual+sess):

|                                          |   gain_share |
|:-----------------------------------------|-------------:|
| sess_prev1_ratio                         |        0.472 |
| sess_prev3_ratio                         |        0.196 |
| sess_rec_full_rate                       |        0.050 |
| rate__user__full_rec_per_listen_rec__w30 |        0.030 |
| sess_mean_ratio                          |        0.027 |
| sess_n_prev                              |        0.019 |
| sess_early_skip_rate                     |        0.018 |
| rate__user__full_rec_per_listen_rec__w7  |        0.013 |
| rate__user__full_per_listen__w30         |        0.012 |
| rate__user__full_rec_per_listen_rec__all |        0.010 |
| rate__user__early_skip_per_listen__w7    |        0.008 |
| rate__user_album__full_per_listen__all   |        0.008 |
| rate__user_artist__full_per_listen__all  |        0.007 |
| rate__user_item__full_per_listen__all    |        0.007 |
| sess_full_rate                           |        0.007 |

Топ-15 признаков по gain (autofe+sess):

|                                                  |   gain_share |
|:-------------------------------------------------|-------------:|
| sess_prev1_ratio                                 |        0.451 |
| sess_prev3_ratio                                 |        0.188 |
| sess_rec_full_rate                               |        0.078 |
| drate__user__full_rec_per_listen_rec__h3         |        0.023 |
| sess_early_skip_rate                             |        0.019 |
| sess_n_prev                                      |        0.019 |
| sess_mean_ratio                                  |        0.018 |
| drate__user__full_rec_per_listen_rec__h30        |        0.015 |
| drate__user__ratio_sum_per_listen__h14           |        0.009 |
| ctx_hour                                         |        0.008 |
| rate__user__early_skip_per_listen__w14           |        0.007 |
| drate__user__early_skip_per_listen__h14          |        0.006 |
| drate__user_album__ratio_sum_per_listen__h90     |        0.006 |
| rate__item__full_rec_per_listen_rec__w180        |        0.006 |
| drate__user_artist__full_rec_per_listen_rec__h90 |        0.006 |

Топ-15 признаков по gain (autofe_full):

|                                           |   gain_share |
|:------------------------------------------|-------------:|
| sg__any__ratio__k5                        |        0.340 |
| sg__day__ratio__k5                        |        0.155 |
| sg__day__frec__k5                         |        0.125 |
| sg__any__frec__k5                         |        0.047 |
| sg__day__frec__k10                        |        0.034 |
| sg__any__ratio__k10                       |        0.031 |
| drate__user__full_rec_per_listen_rec__h3  |        0.024 |
| drate__user__full_rec_per_listen_rec__h30 |        0.015 |
| sg__day__ratio__k10                       |        0.010 |
| sg__any__frec__k10                        |        0.010 |
| sg__any__esk__k20                         |        0.007 |
| sp__item__gap                             |        0.007 |
| sg__day__n__kall                          |        0.007 |
| drate__user__full_per_listen__h14         |        0.006 |
| ctx_hour                                  |        0.005 |

## u5000

data: {"events": 23636943, "users": 5000, "items": 649741, "split": {"train": [266, 286], "val": [287, 293], "test": [294, 300]}, "rows": {"train": 1006214, "val": 348789, "test": 348364}, "pos_rate": {"train": 0.6709984158439457, "val": 0.6895773662586836, "test": 0.6992169110470657}, "share_rec_listens": 0.47958611415639985}

timings (s): {"load_s": 24.1, "session_s": 6.0, "session_grammar_s": 35.1, "engine_build_s": 45.4, "manual_features_s": 28.2, "auto_features_s": 42.5, "autofe_full_features_s": 93.4, "peak_rss_gb": 9.7}

| model                       |   n_feat |   val_logloss |   val_ndcg@10 |   test_logloss |   test_auc |   test_gauc |   test_ndcg@10 |    fit_s |
|:----------------------------|---------:|--------------:|--------------:|---------------:|-----------:|------------:|---------------:|---------:|
| pop_item_fullrate           |        1 |        0.7396 |        0.7226 |         0.7318 |     0.6462 |      0.5437 |         0.7269 | nan      |
| user_fullrate               |        1 |        0.4843 |        0.7064 |         0.4752 |     0.8136 |      0.5000 |         0.7175 | nan      |
| ctx_only                    |        3 |        0.6156 |        0.7296 |         0.6073 |     0.5681 |      0.5433 |         0.7351 |   2.4234 |
| manual                      |       58 |        0.4506 |        0.7648 |         0.4410 |     0.8389 |      0.5912 |         0.7715 |  16.0987 |
| autofe                      |       48 |        0.4497 |        0.7678 |         0.4416 |     0.8383 |      0.5906 |         0.7708 |  13.0150 |
| manual+sess                 |       66 |        0.3526 |        0.7758 |         0.3427 |     0.9093 |      0.6668 |         0.7805 |  22.2862 |
| random_sameN+sess           |       66 |        0.3521 |        0.7779 |         0.3428 |     0.9093 |      0.6676 |         0.7806 |  25.6269 |
| manual_retuned_windows+sess |       48 |        0.3526 |        0.7758 |         0.3430 |     0.9091 |      0.6646 |         0.7789 |  19.4027 |
| autofe+sess                 |       56 |        0.3519 |        0.7778 |         0.3425 |     0.9094 |      0.6676 |         0.7812 |  18.8496 |
| autofe_sameN+sess           |       66 |        0.3519 |        0.7781 |         0.3426 |     0.9094 |      0.6675 |         0.7809 |  22.2110 |
| manual∪autofe+sess          |      110 |        0.3516 |        0.7804 |         0.3422 |     0.9096 |      0.6711 |         0.7842 |  38.2928 |
| autofe_full                 |      103 |        0.3581 |        0.7681 |         0.3480 |     0.9061 |      0.6606 |         0.7738 |  34.8847 |
| autofe_full+sess            |      111 |        0.3474 |        0.7822 |         0.3379 |     0.9119 |      0.6756 |         0.7845 |  38.9391 |
| manual+sess_lambdarank      |       66 |        0.9847 |        0.8420 |         0.9907 |     0.5481 |      0.7173 |         0.8461 |  72.0668 |
| autofe+sess_lambdarank      |       56 |        1.0145 |        0.8426 |         1.0250 |     0.5405 |      0.7180 |         0.8460 |  83.5423 |
| autofe_full_lambdarank      |      103 |        0.6375 |        0.8139 |         0.6339 |     0.7844 |      0.6848 |         0.8170 | 100.6177 |

Paired bootstrap по пользователям, разница с manual+sess (test):

| model                       | gauc                       | logloss                    | ndcg@10                    |
|:----------------------------|:---------------------------|:---------------------------|:---------------------------|
| autofe                      | -0.0762 [-0.0845, -0.0684] | +0.0988 [+0.0911, +0.1068] | -0.0097 [-0.0143, -0.0051] |
| autofe+sess                 | +0.0008 [-0.0020, +0.0035] | -0.0002 [-0.0006, +0.0002] | +0.0007 [-0.0009, +0.0023] |
| autofe+sess_lambdarank      | +0.0513 [+0.0439, +0.0591] | +0.6822 [+0.6416, +0.7296] | +0.0655 [+0.0623, +0.0691] |
| autofe_full                 | -0.0061 [-0.0097, -0.0023] | +0.0053 [+0.0043, +0.0063] | -0.0067 [-0.0091, -0.0040] |
| autofe_full+sess            | +0.0088 [+0.0056, +0.0122] | -0.0049 [-0.0055, -0.0043] | +0.0040 [+0.0019, +0.0061] |
| autofe_full_lambdarank      | +0.0180 [+0.0133, +0.0231] | +0.2912 [+0.2788, +0.3045] | +0.0365 [+0.0331, +0.0398] |
| autofe_sameN+sess           | +0.0007 [-0.0015, +0.0029] | -0.0001 [-0.0005, +0.0003] | +0.0003 [-0.0012, +0.0020] |
| ctx_only                    | -0.1234 [-0.1324, -0.1146] | +0.2645 [+0.2539, +0.2756] | -0.0454 [-0.0509, -0.0401] |
| manual                      | -0.0756 [-0.0837, -0.0679] | +0.0983 [+0.0908, +0.1064] | -0.0091 [-0.0133, -0.0049] |
| manual+sess                 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| manual+sess_lambdarank      | +0.0506 [+0.0434, +0.0578] | +0.6480 [+0.6110, +0.6907] | +0.0656 [+0.0622, +0.0687] |
| manual_retuned_windows+sess | -0.0021 [-0.0045, -0.0001] | +0.0003 [-0.0001, +0.0006] | -0.0016 [-0.0031, +0.0000] |
| manual∪autofe+sess          | +0.0044 [+0.0020, +0.0067] | -0.0006 [-0.0009, -0.0002] | +0.0037 [+0.0022, +0.0052] |
| random_sameN+sess           | +0.0009 [-0.0016, +0.0034] | +0.0000 [-0.0004, +0.0004] | +0.0001 [-0.0015, +0.0017] |

Топ-15 признаков по gain (manual+sess):

|                                          |   gain_share |
|:-----------------------------------------|-------------:|
| sess_prev1_ratio                         |        0.469 |
| sess_prev3_ratio                         |        0.225 |
| sess_rec_full_rate                       |        0.062 |
| rate__user__full_rec_per_listen_rec__w30 |        0.031 |
| sess_mean_ratio                          |        0.024 |
| sess_n_prev                              |        0.016 |
| rate__user__full_rec_per_listen_rec__all |        0.016 |
| rate__user_item__full_per_listen__all    |        0.014 |
| sess_early_skip_rate                     |        0.012 |
| rate__user__full_rec_per_listen_rec__w7  |        0.011 |
| rate__user__full_per_listen__w30         |        0.008 |
| sess_full_rate                           |        0.008 |
| rate__user_artist__full_per_listen__w30  |        0.007 |
| rate__user__early_skip_per_listen__w7    |        0.006 |
| rate__user_artist__full_per_listen__all  |        0.006 |

Топ-15 признаков по gain (autofe+sess):

|                                                  |   gain_share |
|:-------------------------------------------------|-------------:|
| sess_prev1_ratio                                 |        0.442 |
| sess_prev3_ratio                                 |        0.235 |
| sess_rec_full_rate                               |        0.099 |
| drate__user__full_rec_per_listen_rec__h30        |        0.029 |
| drate__user__full_rec_per_listen_rec__h3         |        0.023 |
| sess_n_prev                                      |        0.016 |
| sess_mean_ratio                                  |        0.016 |
| sess_early_skip_rate                             |        0.011 |
| drate__user_item__ratio_sum_per_listen__h90      |        0.010 |
| drate__user_artist__full_rec_per_listen_rec__h90 |        0.006 |
| drate__user__early_skip_per_listen__h14          |        0.006 |
| dcnt__user_item__full__h30                       |        0.006 |
| drate__user_album__ratio_sum_per_listen__h90     |        0.006 |
| ctx_hour                                         |        0.005 |
| rate__item__full_rec_per_listen_rec__w180        |        0.005 |

Топ-15 признаков по gain (autofe_full):

|                                             |   gain_share |
|:--------------------------------------------|-------------:|
| sg__any__ratio__k5                          |        0.364 |
| sg__day__ratio__k5                          |        0.136 |
| sg__any__frec__k5                           |        0.112 |
| sg__day__frec__k5                           |        0.064 |
| sg__day__frec__k10                          |        0.058 |
| drate__user__full_rec_per_listen_rec__h30   |        0.030 |
| sg__any__ratio__k10                         |        0.025 |
| sg__any__frec__k10                          |        0.019 |
| drate__user__full_rec_per_listen_rec__h3    |        0.019 |
| sg__day__ratio__k10                         |        0.009 |
| sg__any__esk__k20                           |        0.007 |
| sp__item__gap                               |        0.006 |
| drate__user_item__ratio_sum_per_listen__h90 |        0.006 |
| sp__item__any_full                          |        0.005 |
| sg__day__n__kall                            |        0.005 |

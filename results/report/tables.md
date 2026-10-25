## u1000

data: {"events": 4768010, "users": 1000, "items": 279380, "split": {"train": [266, 286], "val": [287, 293], "test": [294, 300]}, "rows": {"train": 211850, "val": 74606, "test": 71659}, "pos_rate": {"train": 0.6658531980174652, "val": 0.6677210948181111, "test": 0.6861803820873861}, "share_rec_listens": 0.4852954486892164}

timings (s): {"load_s": 15.7, "engine_build_s": 8.0, "session_s": 1.2, "manual_features_s": 4.3, "autofe_search_s": 223.5, "autofe_stage1_s": 169.1, "autofe_stage2_s": 54.4, "autofe_n_candidates": 2401, "autofe_n_pool": 250, "autofe_best_n": 60, "auto_features_s": 9.2, "peak_rss_gb": 4.8}

| model                       |   n_feat |   val_logloss |   val_ndcg@10 |   test_logloss |   test_auc |   test_gauc |   test_ndcg@10 |    fit_s |
|:----------------------------|---------:|--------------:|--------------:|---------------:|-----------:|------------:|---------------:|---------:|
| pop_item_fullrate           |        1 |        0.9831 |        0.7142 |         0.9478 |     0.6372 |      0.5368 |         0.7209 | nan      |
| user_fullrate               |        1 |        0.4822 |        0.6982 |         0.4708 |     0.8228 |      0.5000 |         0.7130 | nan      |
| ctx_only                    |        3 |        0.6295 |        0.7198 |         0.6167 |     0.5688 |      0.5333 |         0.7281 |   1.8112 |
| manual                      |       58 |        0.4550 |        0.7531 |         0.4507 |     0.8379 |      0.5761 |         0.7572 |   4.3821 |
| autofe                      |       63 |        0.4531 |        0.7577 |         0.4466 |     0.8419 |      0.5795 |         0.7553 |   4.6712 |
| manual+sess                 |       66 |        0.3487 |        0.7748 |         0.3469 |     0.9101 |      0.6592 |         0.7719 |   5.5622 |
| random_sameN+sess           |       66 |        0.3485 |        0.7723 |         0.3465 |     0.9103 |      0.6589 |         0.7706 |   6.0435 |
| manual_retuned_windows+sess |       48 |        0.3490 |        0.7737 |         0.3459 |     0.9107 |      0.6622 |         0.7759 |   5.0609 |
| autofe+sess                 |       71 |        0.3477 |        0.7770 |         0.3457 |     0.9108 |      0.6645 |         0.7758 |   5.9212 |
| autofe_sameN+sess           |       66 |        0.3479 |        0.7766 |         0.3457 |     0.9108 |      0.6653 |         0.7761 |   6.0428 |
| manual∪autofe+sess          |      124 |        0.3479 |        0.7748 |         0.3456 |     0.9109 |      0.6626 |         0.7743 |   7.6286 |
| manual+sess_lambdarank      |       66 |        0.8069 |        0.8344 |         0.8192 |     0.6512 |      0.7016 |         0.8287 |  12.9600 |
| autofe+sess_lambdarank      |       71 |        0.8314 |        0.8302 |         0.8458 |     0.6471 |      0.7060 |         0.8324 |  14.9759 |

Paired bootstrap по пользователям, разница с manual+sess (test):

| model                       | gauc                       | logloss                    | ndcg@10                    |
|:----------------------------|:---------------------------|:---------------------------|:---------------------------|
| autofe                      | -0.0797 [-0.0970, -0.0622] | +0.0997 [+0.0846, +0.1158] | -0.0165 [-0.0279, -0.0050] |
| autofe+sess                 | +0.0052 [+0.0009, +0.0097] | -0.0012 [-0.0023, -0.0002] | +0.0039 [-0.0004, +0.0082] |
| autofe+sess_lambdarank      | +0.0468 [+0.0352, +0.0606] | +0.4989 [+0.4448, +0.5514] | +0.0605 [+0.0532, +0.0677] |
| autofe_sameN+sess           | +0.0060 [+0.0013, +0.0117] | -0.0013 [-0.0023, -0.0002] | +0.0042 [+0.0001, +0.0084] |
| ctx_only                    | -0.1259 [-0.1466, -0.1056] | +0.2698 [+0.2481, +0.2939] | -0.0437 [-0.0568, -0.0304] |
| manual                      | -0.0831 [-0.1018, -0.0637] | +0.1038 [+0.0890, +0.1200] | -0.0147 [-0.0261, -0.0028] |
| manual+sess                 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| manual+sess_lambdarank      | +0.0423 [+0.0307, +0.0549] | +0.4722 [+0.4221, +0.5215] | +0.0568 [+0.0501, +0.0637] |
| manual_retuned_windows+sess | +0.0030 [-0.0004, +0.0067] | -0.0010 [-0.0019, -0.0002] | +0.0040 [+0.0003, +0.0077] |
| manual∪autofe+sess          | +0.0034 [-0.0005, +0.0077] | -0.0013 [-0.0022, -0.0005] | +0.0024 [-0.0011, +0.0057] |
| random_sameN+sess           | -0.0004 [-0.0060, +0.0058] | -0.0004 [-0.0016, +0.0006] | -0.0013 [-0.0053, +0.0029] |

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

Распределение лучших масштабов: {"all/other": 50, "h=90.0": 30, "h=30.0": 13, "h=14.0": 12, "w=60.0": 7, "h=3.0": 6, "w=1.0": 6, "w=90.0": 5, "h=7.0": 5, "w=7.0": 4, "h=1.0": 4, "w=30.0": 3, "w=21.0": 3, "w=2.0": 2, "w=45.0": 2, "w=3.0": 2, "w=120.0": 1, "w=14.0": 1}

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
| sess_prev1_ratio                                 |        0.459 |
| sess_prev3_ratio                                 |        0.182 |
| sess_rec_full_rate                               |        0.072 |
| sess_early_skip_rate                             |        0.023 |
| drate__user__full_rec_per_listen_rec__h3         |        0.021 |
| sess_n_prev                                      |        0.018 |
| drate__user__full_rec_per_listen_rec__h30        |        0.015 |
| sess_mean_ratio                                  |        0.014 |
| drate__user__early_skip_per_listen__h14          |        0.008 |
| drate__user__ratio_sum_per_listen__h14           |        0.007 |
| ctx_hour                                         |        0.007 |
| rate__user__early_skip_per_listen__w14           |        0.006 |
| sess_full_rate                                   |        0.006 |
| drate__user_artist__full_rec_per_listen_rec__h90 |        0.005 |
| drate__user__full_per_listen__h14                |        0.005 |

## u5000

data: {"events": 23636943, "users": 5000, "items": 649741, "split": {"train": [266, 286], "val": [287, 293], "test": [294, 300]}, "rows": {"train": 1006214, "val": 348789, "test": 348364}, "pos_rate": {"train": 0.6709984158439457, "val": 0.6895773662586836, "test": 0.6992169110470657}, "share_rec_listens": 0.47958611415639985}

timings (s): {"load_s": 23.3, "engine_build_s": 47.8, "session_s": 5.6, "manual_features_s": 28.7, "auto_features_s": 74.1, "peak_rss_gb": 9.8}

| model                       |   n_feat |   val_logloss |   val_ndcg@10 |   test_logloss |   test_auc |   test_gauc |   test_ndcg@10 |    fit_s |
|:----------------------------|---------:|--------------:|--------------:|---------------:|-----------:|------------:|---------------:|---------:|
| pop_item_fullrate           |        1 |        0.7396 |        0.7226 |         0.7318 |     0.6462 |      0.5437 |         0.7269 | nan      |
| user_fullrate               |        1 |        0.4843 |        0.7064 |         0.4752 |     0.8136 |      0.5000 |         0.7175 | nan      |
| ctx_only                    |        3 |        0.6156 |        0.7296 |         0.6073 |     0.5681 |      0.5433 |         0.7351 |   2.2171 |
| manual                      |       58 |        0.4506 |        0.7648 |         0.4410 |     0.8389 |      0.5912 |         0.7715 |  16.7611 |
| autofe                      |       63 |        0.4497 |        0.7678 |         0.4415 |     0.8384 |      0.5942 |         0.7710 |  16.9723 |
| manual+sess                 |       66 |        0.3526 |        0.7758 |         0.3427 |     0.9093 |      0.6668 |         0.7805 |  23.1928 |
| random_sameN+sess           |       66 |        0.3526 |        0.7750 |         0.3434 |     0.9089 |      0.6658 |         0.7796 |  25.8550 |
| manual_retuned_windows+sess |       48 |        0.3530 |        0.7754 |         0.3433 |     0.9090 |      0.6636 |         0.7778 |  18.5785 |
| autofe+sess                 |       71 |        0.3518 |        0.7779 |         0.3426 |     0.9094 |      0.6705 |         0.7826 |  24.7841 |
| autofe_sameN+sess           |       66 |        0.3520 |        0.7788 |         0.3425 |     0.9095 |      0.6695 |         0.7818 |  22.7979 |
| manual∪autofe+sess          |      124 |        0.3515 |        0.7804 |         0.3421 |     0.9097 |      0.6691 |         0.7840 |  40.5032 |
| manual+sess_lambdarank      |       66 |        0.9847 |        0.8420 |         0.9907 |     0.5481 |      0.7173 |         0.8461 |  72.9424 |
| autofe+sess_lambdarank      |       71 |        0.9754 |        0.8429 |         0.9838 |     0.5538 |      0.7178 |         0.8466 |  67.6527 |

Paired bootstrap по пользователям, разница с manual+sess (test):

| model                       | gauc                       | logloss                    | ndcg@10                    |
|:----------------------------|:---------------------------|:---------------------------|:---------------------------|
| autofe                      | -0.0726 [-0.0808, -0.0644] | +0.0987 [+0.0911, +0.1069] | -0.0095 [-0.0140, -0.0052] |
| autofe+sess                 | +0.0038 [+0.0009, +0.0064] | -0.0002 [-0.0006, +0.0002] | +0.0021 [+0.0005, +0.0037] |
| autofe+sess_lambdarank      | +0.0510 [+0.0441, +0.0584] | +0.6410 [+0.6027, +0.6853] | +0.0661 [+0.0629, +0.0697] |
| autofe_sameN+sess           | +0.0027 [+0.0003, +0.0049] | -0.0003 [-0.0006, +0.0001] | +0.0012 [-0.0004, +0.0028] |
| ctx_only                    | -0.1234 [-0.1324, -0.1146] | +0.2645 [+0.2539, +0.2756] | -0.0454 [-0.0509, -0.0401] |
| manual                      | -0.0756 [-0.0837, -0.0679] | +0.0983 [+0.0908, +0.1064] | -0.0091 [-0.0133, -0.0049] |
| manual+sess                 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| manual+sess_lambdarank      | +0.0506 [+0.0434, +0.0578] | +0.6480 [+0.6110, +0.6907] | +0.0656 [+0.0622, +0.0687] |
| manual_retuned_windows+sess | -0.0031 [-0.0053, -0.0011] | +0.0005 [+0.0002, +0.0009] | -0.0027 [-0.0043, -0.0011] |
| manual∪autofe+sess          | +0.0024 [+0.0002, +0.0044] | -0.0006 [-0.0009, -0.0003] | +0.0035 [+0.0019, +0.0049] |
| random_sameN+sess           | -0.0009 [-0.0035, +0.0013] | +0.0007 [+0.0002, +0.0012] | -0.0009 [-0.0028, +0.0009] |

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
| sess_prev1_ratio                                 |        0.443 |
| sess_prev3_ratio                                 |        0.228 |
| sess_rec_full_rate                               |        0.093 |
| drate__user__full_rec_per_listen_rec__h30        |        0.029 |
| drate__user__full_rec_per_listen_rec__h3         |        0.022 |
| sess_early_skip_rate                             |        0.016 |
| sess_n_prev                                      |        0.016 |
| sess_mean_ratio                                  |        0.012 |
| drate__user_item__ratio_sum_per_listen__h90      |        0.008 |
| drate__user_artist__full_rec_per_listen_rec__h90 |        0.006 |
| drate__user__early_skip_per_listen__h14          |        0.006 |
| drate__user_album__ratio_sum_per_listen__h90     |        0.005 |
| ctx_hour                                         |        0.005 |
| dcnt__user_item__full__h30                       |        0.005 |
| rate__user_artist__full_rec_per_listen_rec__w120 |        0.004 |

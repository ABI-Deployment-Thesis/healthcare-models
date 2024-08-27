## Fetal Health

### Python 3.9.6

| Library      | Version | Required |
| ------------ | ------- | -------- |
| joblib       | 1.0.1   | X        |
| numpy        | 1.22.4  | X        |
| pandas       | 1.5.3   | -        |
| scikit-learn | 0.24.1  | X        |

| Feature                                                | Type  | Example |
| ------------------------------------------------------ | ----- | ------- |
| baseline value                                         | float | 133.0   |
| accelerations                                          | float | 0.002   |
| fetal_movement                                         | float | 0.01    |
| uterine_contractions                                   | float | 0.003   |
| light_decelerations                                    | float | 0.002   |
| severe_decelerations                                   | float | 0.0     |
| prolongued_decelerations                               | float | 0.0     |
| abnormal_short_term_variability                        | float | 46.0    |
| mean_value_of_short_term_variability                   | float | 1.1     |
| percentage_of_time_with_abnormal_long_term_variability | float | 0.0     |
| mean_value_of_long_term_variability                    | float | 15.4    |
| histogram_width                                        | float | 69.0    |
| histogram_min                                          | float | 95.0    |
| histogram_max                                          | float | 164.0   |
| histogram_number_of_peaks                              | float | 5.0     |
| histogram_number_of_zeroes                             | float | 0.0     |
| histogram_mode                                         | float | 139.0   |
| histogram_mean                                         | float | 135.0   |
| histogram_median                                       | float | 138.0   |
| histogram_variance                                     | float | 9.0     |
| histogram_tendency                                     | float | 0.0     |

**Accuracy:** 0.9467

---

### R 4.1.3

| Library      | Version | Required |
| ------------ | ------- | -------- |
| randomForest | 4.7.1   | X        |

| Feature                                                | Type  | Example |
| ------------------------------------------------------ | ----- | ------- |
| baseline.value                                         | float | 132     |
| accelerations                                          | float | 0.006   |
| fetal_movement                                         | float | 0.0     |
| uterine_contractions                                   | float | 0.006   |
| light_decelerations                                    | float | 0.003   |
| severe_decelerations                                   | float | 0       |
| prolongued_decelerations                               | float | 0.0     |
| abnormal_short_term_variability                        | float | 17      |
| mean_value_of_short_term_variability                   | float | 2.1     |
| percentage_of_time_with_abnormal_long_term_variability | float | 0       |
| mean_value_of_long_term_variability                    | float | 10.4    |
| histogram_width                                        | float | 130     |
| histogram_min                                          | float | 68      |
| histogram_max                                          | float | 198     |
| histogram_number_of_peaks                              | float | 6       |
| histogram_number_of_zeroes                             | float | 1       |
| histogram_mode                                         | float | 141     |
| histogram_mean                                         | float | 136     |
| histogram_median                                       | float | 140     |
| histogram_variance                                     | float | 12      |
| histogram_tendency                                     | float | 0       |

**Accuracy:** 0.9350
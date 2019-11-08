import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from Models import calc_prob_multiple_samples, diff_features
from generate_samples import generate_samples, generate_multiple_samples

n_samples = 10000
n_contributors = 3
threshold = 50

thresholds = list(range(0, 501, 50))
num_loci = list(range(1, 21))

results = defaultdict(dict)

if __name__ == "__main__":
    for n in num_loci:
        np.random.seed(42)
        random.seed(42)

        shuffle_idx = np.random.choice(np.arange(0, n_samples, 1),
                                       replace=False, size=n_samples)

        # generate binary data
        X_train, y_train = generate_samples(n_samples, True,
                                            threshold=threshold,
                                            n_contributors=n_contributors)
        X_train, y_train = X_train[shuffle_idx, :], y_train[shuffle_idx]

        X_test, y_test = generate_samples(1000, True, threshold=threshold,
                                          n_contributors=n_contributors)

        X_extra = generate_multiple_samples(1000, True, threshold=threshold,
                                            n_contributors=n_contributors,
                                            n_loci=n)

        diff_X_train = diff_features(X_train)
        diff_X_test = diff_features(X_test)

        # create features
        # baseline_X_train = baseline(X_train)
        # baseline_X_test = baseline(X_test)

        for model in [('Logistic', LogisticRegression()),
                      ('RF', RandomForestClassifier()),
                      ('SVC', SVC(probability=True))]:
            # mod = model[1]
            # mod.fit(X_train, y_train)
            # y_hat = mod.predict(X_test)
            # accuracy_normal = np.mean(y_hat == y_test)
            # print("Accuracy normal data", model[0], accuracy_normal)

            # mod = model[1]
            # mod.fit(baseline_X_train, y_train)
            # y_hat = mod.predict(baseline_X_test)
            # accuracy_baseline = np.mean(y_hat == y_test)
            # print("Accuracy baseline data", model[0], accuracy_baseline)

            mod = model[1]
            mod.fit(diff_X_train, y_train)
            # y_hat = mod.predict(diff_X_test)
            # accuracy_diff = np.mean(y_hat == y_test)
            # print("Accuracy diff data", model[0], accuracy_diff)

            # results[model[0]][threshold] = dict(
            #     normal=accuracy_normal,
            #     # baseline=accuracy_baseline,
            #     # diff=accuracy_diff
            # )

            y_hat_extra = np.mean(calc_prob_multiple_samples(X_extra, model[1]),
                                  axis=0)

            accuracy = accuracy_score(y_test,
                                      [1 if m > .5 else 0 for m in y_hat_extra])
            results[model[0]][n] = dict(diff=accuracy)

        # baseline model
        # y_baseline_1 = (baseline_X_test[:, 0] < np.mean(baseline_X_train[:, 0])) * 1
        # baseline_acc_1 = np.mean(y_baseline_1 == y_test)
        # results['Baseline'][threshold] = dict(baseline=baseline_acc_1)

fig, ax = plt.subplots(1, 1)
ax.set_xlabel('#Loci')
ax.set_ylabel('Accuracy')
for model, r1 in results.items():
    thresholds = list(r1.keys())
    for feature_type in ['normal', 'diff', 'baseline']:
        try:
            accuracies = [k[feature_type] for k in r1.values()]
            line, *_ = ax.plot(thresholds, accuracies)
            line.set_label(f'{model} ({feature_type})')
        except (KeyError, IndexError):
            pass
ax.legend(bbox_to_anchor=(0, 1), loc='upper right', ncol=1)
plt.show()
print(results)

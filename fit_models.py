import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from Models import baseline, diff_features
from generate_samples import generate_samples
from matplotlib import pyplot as plt

n_samples = 10000
threshold = 0
n_contributors = 3

if __name__ == "__main__":
    shuffle_idx = np.random.choice(np.arange(0, n_samples, 1), replace=False, size=n_samples)

    # # generate binary data
    X_train, y_train = generate_samples(n_samples, True, threshold=threshold, n_contributors=n_contributors)


    X_train, y_train = X_train[shuffle_idx, :], y_train[shuffle_idx]

    X_test, y_test = generate_samples(1000, True, threshold=threshold, n_contributors=n_contributors)
    X_extra, y_extra = generate_samples(1000, True, threshold=threshold, n_contributors=n_contributors)

    diff_X_train = diff_features(X_train)
    diff_X_test = diff_features(X_test)
    diff_X_test_extra = diff_features(X_test)

    # create features
    baseline_X_train = baseline(X_train)
    baseline_X_test = baseline(X_test)
    fig, axes = plt.subplots(1, 3)

    # verschillende dropouts
    thresholds = np.arange(0, 100, 1)


    for i, model in enumerate([('Logistic: ', LogisticRegression()), ('RF: ', RandomForestClassifier()),
                  ('SVM: ', SVC(probability=True))]):

        mod = model[1]
        mod.fit(diff_X_train, y_train)

        ##### deze wil je jeroen #######
        y_baseline_1 = (baseline_X_test[:, 0] < np.mean(baseline_X_train[:, 0])) * 1
        ###########################################################################
        y_baseline_2 = (baseline_X_test[:, 1] < np.mean(baseline_X_train[:, 1])) * 1

        baseline_acc_1 = np.mean(y_baseline_1 == y_test)
        baseline_acc_2 = np.mean(y_baseline_2 == y_test)

        y_hat = mod.predict_proba(diff_X_test)[:, 1]
        y_hat_extra = np.mean( np.concatenate((mod.predict_proba(diff_X_test_extra), y_hat)),axis=1)

        fpr, tpr, thresholds = roc_curve(y_test, y_hat)
        accuracy_scores = []
        for thresh in thresholds:
            accuracy_scores.append(accuracy_score(y_test, [1 if m > thresh else 0 for m in y_hat]))


        axes[i].plot(thresholds, tpr, color='green')
        axes[i].plot(thresholds, fpr, color='red')
        axes[i].plot(thresholds, accuracy_scores, color='black')
        axes[i].plot(thresholds, np.repeat(baseline_acc_1, len(thresholds)), color='yellow')
        axes[i].plot(thresholds, np.repeat(baseline_acc_2, len(thresholds)), color='orange')

    plt.show()

    # accuracies = []
    # sample_sizes = np.logspace(2, np.log10(n_samples), num=10)
    # mod = LogisticRegression()
    # mod.fit(baseline_X_train, y_train)
    # y_hat = mod.predict_proba(baseline_X_test)[:, 1]
    # fpr, tpr, thresholds = roc_curve(y_test, y_hat)

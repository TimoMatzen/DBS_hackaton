import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from Models import baseline, diff_features
from generate_samples import generate_samples
from matplotlib import pyplot as plt

n_samples = 100000
n_contributors = 2
if __name__ == "__main__":
    shuffle_idx = np.random.choice(np.arange(0, n_samples, 1), replace=False, size=n_samples)

    # generate binary data
    X_train, y_train = generate_samples(n_samples, True, threshold=0, n_contributors=n_contributors)
    X_train, y_train = X_train[shuffle_idx, :], y_train[shuffle_idx]

    X_test, y_test = generate_samples(1000, True, threshold=0, n_contributors=n_contributors)

    diff_X_train = diff_features(X_train)
    diff_X_test = diff_features(X_test)
    # create features
    baseline_X_train = baseline(X_train)
    baseline_X_test = baseline(X_test)

    # for model in [('Logistic: ', LogisticRegression()), ('RF: ', RandomForestClassifier()), ('SVM: ', SVC())]:
    #
    #     # mod = model[1]
    #     # mod.fit(X_train, y_train)
    #     # y_hat = mod.predict(X_test)
    #     #
    #     # print("Accuracy normal data", model[0], np.mean(y_hat == y_test))
    #     #
    #     # mod = model[1]
    #     # mod.fit(baseline_X_train, y_train)
    #     # y_hat = mod.predict(baseline_X_test)
    #     #
    #     # print("Accuracy baseline data", model[0], np.mean(y_hat == y_test))
    #
    #     mod = model[1]
    #     # mod.fit(diff_X_train, y_train)
    #     # y_hat = mod.predict(diff_X_test)
    #     #
    #     GridSearchCV(mod, parameters)
    #
    #     print("Accuracy diff data", model[0], np.mean(y_hat == y_test))
    accuracies = []
    sample_sizes = np.logspace(2, np.log10(n_samples), num=10)
    mod = LogisticRegression()
    mod.fit(baseline_X_train, y_train)
    y_hat = mod.predict(baseline_X_test)
    print(np.mean(y_test== y_hat))
    # for size in sample_sizes:
    #     print(size)
    #     mod = RandomForestClassifier()
    #     mod.fit(diff_X_train[:int(size), :], y_train[:int(size)])
    #     y_hat = mod.predict_proba(diff_X_test)
    #
    #     accuracies.append(np.mean(y_hat == y_test))


    plt.scatter(sample_sizes, accuracies)
    plt.show()
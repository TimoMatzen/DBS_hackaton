import numpy as np

def baseline(df, threshold=5):
    # Welke features zijn hetzelfde
    same = df[:, 0:10] == df[:, 10:]

    # sommeer over dezelfde features
    sum_features = np.sum(same, axis=0)

    # wanneer boven vijf return 1
    predictions = sum_features > threshold

    return(predictions)
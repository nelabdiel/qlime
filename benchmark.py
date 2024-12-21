import time
import random
import csv
import itertools
import json
import re

import numpy as np
import pennylane as qml
from concurrent.futures import ThreadPoolExecutor

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Classical LIME
from lime.lime_text import LimeTextExplainer

##############################################################################
# PART 0: DATA LOADING AND PREPROCESSING
##############################################################################

def clean_text(text):
    """
    Removes HTML tags and converts to lowercase.
    """
    # Remove anything between <...> tags, then lowercase the text
    cleaned = re.sub(r'<.*?>', '', text).lower()
    return cleaned

def load_imdb_subset(
    num_samples=500, 
    min_df=1, 
    max_features=10, 
    stopwords_option=True
):
    """
    Loads a subset of IMDb data, returns:
      - X_train, X_test (lists of text)
      - y_train, y_test (0/1 sentiment)
      - vectorizer (CountVectorizer)
    
    Now with text cleaning for HTML, lowercase, etc.
    """
    data = load_files(
        "./aclImdb/train", 
        categories=['pos','neg'], 
        encoding="utf-8", 
        decode_error="replace"
    )
    X_text_all, y_all = data.data, data.target

    # Clean text (HTML removal + lowercase)
    X_text_all = [clean_text(txt) for txt in X_text_all]

    # Shuffle & truncate to num_samples
    full_idx = np.arange(len(X_text_all))
    np.random.shuffle(full_idx)
    subset_idx = full_idx[:num_samples]
    X_text = [X_text_all[i] for i in subset_idx]
    y = y_all[subset_idx]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42
    )

    # Vectorizer: presence/absence
    if stopwords_option:
        vectorizer = CountVectorizer(
            binary=True, stop_words='english', 
            min_df=min_df, max_features=max_features
        )
    else:
        vectorizer = CountVectorizer(
            binary=True, stop_words=None, 
            min_df=min_df, max_features=max_features
        )

    vectorizer.fit(X_train)
    return X_train, X_test, y_train, y_test, vectorizer


def train_logistic_classifier(X_train, y_train, vectorizer):
    """
    Trains a logistic regression on the binary presence/absence of words.
    Returns the fitted model.
    """
    X_train_bow = vectorizer.transform(X_train)
    clf = LogisticRegression()
    clf.fit(X_train_bow, y_train)
    return clf

##############################################################################
# CLASSICAL LIME
##############################################################################

def run_classical_lime(
    text_sample, clf, vectorizer, 
    k_features=5, num_samples=500
):
    """
    Runs classical LIME on a single text instance.
    Returns the top (word, weight) pairs.
    """
    class_names = ["negative", "positive"]
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(texts):
        bow = vectorizer.transform(texts) 
        return clf.predict_proba(bow)

    explanation = explainer.explain_instance(
        text_sample,
        predict_proba,
        num_features=k_features,
        num_samples=num_samples  # e.g. 300 or 500
    )
    return explanation.as_list()  # list of (word, weight)

##############################################################################
# Q-LIME Pi (Flip Only 1->0)
##############################################################################

def classical_classifier(features, weights, bias=0.0):
    """
    Simple logistic: score = bias + dot(features, weights).
    Then return sigmoid.
    """
    score = bias + np.dot(features, weights)
    return 1 / (1 + np.exp(-score))

def encode_and_flip(features, flip_index=None, shots=None):
    """
    Encode features -> quantum circuit.
    FLIP ONLY if bit == 1 at flip_index (1->0).
    """
    num_qubits = len(features)
    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit():
        for i, f in enumerate(features):
            if i == flip_index and f == 1:
                # 1->0 => RY(0)
                theta = 0.0
            else:
                # otherwise normal encoding: f * (pi/2)
                theta = f * (np.pi / 2)
            qml.RY(theta, wires=i)
        return qml.probs(wires=range(num_qubits))

    return circuit()

def sample_state(probabilities):
    """
    Sample an integer state index from the distribution.
    """
    r = random.random()
    cumsum = 0.0
    for idx, p in enumerate(probabilities):
        cumsum += p
        if r <= cumsum:
            return idx
    return len(probabilities) - 1

def measure_and_map_to_classical(features, flip_index=None, shots=None):
    """
    Run the circuit, measure, return a binary array for the top-likelihood state.
    """
    probs = encode_and_flip(features, flip_index=flip_index, shots=shots)
    measured_state = sample_state(probs)
    num_qubits = len(features)
    bin_string = f"{measured_state:0{num_qubits}b}"
    return [int(bit) for bit in bin_string]

def quantum_lime_explanation(
    features, weights, bias=0.0, shots=None
):
    """
    Flip only features that are 1 -> 0.
    Return array of shape (n_features,) with:
       Delta f_k = (original_pred - new_pred).
    """
    original_pred = classical_classifier(features, weights, bias=bias)
    contributions = np.zeros(len(features))

    def flip_and_predict(i):
        new_vec = measure_and_map_to_classical(features, flip_index=i, shots=shots)
        new_pred = classical_classifier(new_vec, weights, bias=bias)
        return original_pred - new_pred

    # Flip only bits that are 1
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(flip_and_predict, i): i
            for i, val in enumerate(features) if val == 1
        }
        for future in futures:
            i = futures[future]
            contributions[i] = future.result()

    return contributions

##############################################################################
# EXPERIMENTAL ROUTINE
##############################################################################

def run_experiment(
    num_samples=500,
    min_df=1,
    max_features=10,
    stopwords_option=True,
    lime_num_samples=300,
    shots=None,
    n_test_explanations=5
):
    """
    1) Load data with given params (includes text cleaning)
    2) Train logistic classifier
    3) Evaluate test accuracy
    4) Pick n_test_explanations random samples
    5) For each, run classical LIME vs. Q-LIME Pi
    6) Return summary stats
    """
    # A) Load data
    X_train, X_test, y_train, y_test, vectorizer = load_imdb_subset(
        num_samples=num_samples,
        min_df=min_df,
        max_features=max_features,
        stopwords_option=stopwords_option
    )
    # B) Train model
    clf = train_logistic_classifier(X_train, y_train, vectorizer)

    # Evaluate
    X_test_bow = vectorizer.transform(X_test)
    test_acc = accuracy_score(y_test, clf.predict(X_test_bow))

    logistic_weights = clf.coef_[0]
    bias = clf.intercept_[0]

    # We'll track times & top-feature overlap
    lime_times = []
    qlime_times = []
    overlaps = []

    # Random samples for explanation
    n_test = len(X_test)
    sample_indices = np.random.choice(n_test, size=n_test_explanations, replace=False)

    for idx in sample_indices:
        text_sample = X_test[idx]

        # 1) Classical LIME
        start_lime = time.time()
        explanation_lime = run_classical_lime(
            text_sample, clf, vectorizer, 
            k_features=5, num_samples=lime_num_samples
        )
        lime_time = time.time() - start_lime
        lime_times.append(lime_time)

        # parse top features
        lime_dict = dict(explanation_lime)
        top_words_lime = sorted(
            lime_dict.keys(),
            key=lambda w: abs(lime_dict[w]),
            reverse=True
        )[:5]

        # 2) Q-LIME Pi
        bow = vectorizer.transform([text_sample])
        bin_features = bow.toarray()[0]

        start_qlime = time.time()
        contributions_qlime = quantum_lime_explanation(
            bin_features, logistic_weights, bias=bias, shots=shots
        )
        qlime_time = time.time() - start_qlime
        qlime_times.append(qlime_time)

        # top 5 (by absolute value)
        nonzero_indices = [
            (i, abs(contributions_qlime[i])) 
            for i in range(len(contributions_qlime))
        ]
        top_indices_qlime = sorted(nonzero_indices, key=lambda x: x[1], reverse=True)[:5]
        top_words_qlime = [
            vectorizer.get_feature_names_out()[i2]
            for (i2, val) in top_indices_qlime
        ]

        # measure overlap
        overlap = set(top_words_lime).intersection(set(top_words_qlime))
        overlaps.append(len(overlap))

    # Summary
    results = {
        "accuracy": round(test_acc, 4),
        "lime_time_avg": round(np.mean(lime_times), 4),
        "qlime_time_avg": round(np.mean(qlime_times), 4),
        "overlap_avg": round(np.mean(overlaps), 4),
    }
    return results

##############################################################################
# MAIN
##############################################################################

if __name__ == "__main__":
    import pandas as pd

    # Parameter grid to systematically vary certain settings
    param_grid = {
        "num_samples": [500],
        "max_features": [15, 10, 5],
        "stopwords_option": [True, False],
        "lime_num_samples": [300],
        # Shots: None => analytic mode, 100 => finite sampling
        "shots": [None, 100],
    }

    combos = list(itertools.product(*param_grid.values()))
    all_results = []

    for combo in combos:
        (num_samples_, max_features_, stopwords_, lime_samps_, shots_) = combo
        
        print("\n==================================")
        print(f"Running experiment with: "
              f"num_samples={num_samples_}, "
              f"max_features={max_features_}, "
              f"stopwords={stopwords_}, "
              f"lime_num_samples={lime_samps_}, "
              f"shots={shots_}")
        
        res = run_experiment(
            num_samples=num_samples_,
            max_features=max_features_,
            stopwords_option=stopwords_,
            lime_num_samples=lime_samps_,
            shots=shots_,
            n_test_explanations=5
        )
        res_row = {
            "num_samples": num_samples_,
            "max_features": max_features_,
            "stopwords": stopwords_,
            "lime_num_samples": lime_samps_,
            "shots": shots_,
            "accuracy": res["accuracy"],
            "lime_time_avg": res["lime_time_avg"],
            "qlime_time_avg": res["qlime_time_avg"],
            "overlap_avg": res["overlap_avg"],
        }
        print("Results =>", res_row)
        all_results.append(res_row)

    # Save results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv("results_expanded_flips.csv", index=False)
    print("\nAll done! Saved results to 'results_expanded_flips.csv'.")

import time
import random
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
    """Remove HTML tags and convert to lowercase."""
    return re.sub(r'<.*?>', '', text).lower()
    
def load_imdb_subset(num_samples=1000, min_df=2, max_features=20):
    """
    Loads a subset of IMDb data, returns:
      - X_train, X_test (lists of text)
      - y_train, y_test (0/1 sentiment)
      - vectorizer (CountVectorizer)
    """
    data = load_files("./aclImdb/train", categories=['pos','neg'],
                      encoding="utf-8", decode_error="replace")
    X_text, y = data.data, data.target

    # Clean text: remove HTML, lowercase
    X_text = [clean_text(txt) for txt in X_text]

    # Shuffle & truncate
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    X_text = [X_text[i] for i in idx]
    y = y[idx]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42
    )

    # Binary presence/absence vectorizer
    vectorizer = CountVectorizer(binary=True, stop_words='english',
                                 min_df=min_df, max_features=max_features)
    vectorizer.fit(X_train)
    return X_train, X_test, y_train, y_test, vectorizer

##############################################################################
# PART 1: TRAIN A SIMPLE SENTIMENT CLASSIFIER
##############################################################################

def train_logistic_classifier(X_train, y_train, vectorizer):
    """Trains logistic regression on the binary presence/absence of words."""
    X_train_bow = vectorizer.transform(X_train)
    clf = LogisticRegression()
    clf.fit(X_train_bow, y_train)
    return clf

##############################################################################
# PART 2: CLASSICAL LIME EXPLANATIONS
##############################################################################

def run_classical_lime(text_sample, clf, vectorizer, k_features=5):
    """
    Runs classical LIME on a single text instance.
    Returns list of (word, weight) pairs.
    """
    class_names = ["negative", "positive"]
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(texts):
        bow = vectorizer.transform(texts)
        return clf.predict_proba(bow)

    explanation = explainer.explain_instance(
        text_sample,
        predict_proba,
        num_features=k_features
    )
    return explanation.as_list()

##############################################################################
# PART 3: Q-LIME Pi (Quantum LIME) Implementation (flip only 1->0)
##############################################################################

def classical_classifier(features, weights):
    """Simple logistic: dot(features, weights), then sigmoid."""
    score = np.dot(features, weights)
    return 1 / (1 + np.exp(-score))

def encode_and_flip(features, flip_index=None):
    """
    Encode features into a quantum circuit.
    Flip only if bit == 1 at flip_index (1->0).
    """
    dev = qml.device("default.qubit", wires=len(features), shots=None)

    @qml.qnode(dev)
    def circuit():
        for i, f in enumerate(features):
            if i == flip_index and f == 1:
                theta = 0.0  # flip 1->0
            else:
                theta = f * (np.pi / 2)
            qml.RY(theta, wires=i)
        return qml.probs(wires=range(len(features)))

    return circuit()

def sample_state(probabilities):
    """Randomly sample an integer state index from a prob distribution."""
    r = random.random()
    cumsum = 0.0
    for idx, p in enumerate(probabilities):
        cumsum += p
        if r <= cumsum:
            return idx
    return len(probabilities) - 1

def measure_and_map_to_classical(features, flip_index=None):
    """Measure the quantum state, get a single classical bitstring."""
    probs = encode_and_flip(features, flip_index)
    measured_state = sample_state(probs)
    num_qubits = len(features)
    bin_string = f"{measured_state:0{num_qubits}b}"
    return [int(bit) for bit in bin_string]

def quantum_lime_explanation(features, weights):
    """
    For each bit that is 1, flip it to 0, measure difference in classifier output.
    Return array of shape (n_features, ) with contributions.
    """
    original_pred = classical_classifier(features, weights)
    contributions = np.zeros(len(features))

    def flip_and_predict(i):
        new_vec = measure_and_map_to_classical(features, flip_index=i)
        new_pred = classical_classifier(new_vec, weights)
        return original_pred - new_pred

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(flip_and_predict, i): i
            for i, val in enumerate(features) if val == 1
        }
        for fut in futures:
            i = futures[fut]
            contributions[i] = fut.result()

    return contributions

##############################################################################
# PART 4A: EXAMPLE RUN (like your current approach)
##############################################################################

def run_example():
    # 1) Load IMDb subset (small sample)
    X_train, X_test, y_train, y_test, vectorizer = load_imdb_subset(
        num_samples=500,
        min_df=1,
        max_features=10
    )

    # 2) Train logistic regression
    clf = train_logistic_classifier(X_train, y_train, vectorizer)

    # Evaluate
    X_test_bow = vectorizer.transform(X_test)
    test_acc = accuracy_score(y_test, clf.predict(X_test_bow))
    print(f"Test Accuracy = {test_acc:.2f}")

    # Extract logistic weights
    logistic_weights = clf.coef_[0]
    bias = clf.intercept_[0]  # unused in classical_classifier

    # 3) Pick test samples for explanation
    num_samples_to_explain = 5
    sample_indices = np.random.choice(len(X_test), size=num_samples_to_explain, replace=False)

    for idx in sample_indices:
        text_sample = X_test[idx]
        true_label = y_test[idx]

        print("\n=====================================")
        print(f"TEXT SAMPLE (index={idx}): {text_sample}")
        print(f"True Label = {true_label} | Predicted = {clf.predict(vectorizer.transform([text_sample]))[0]}")

        #--- Classical LIME
        start_lime = time.time()
        explanation_lime = run_classical_lime(text_sample, clf, vectorizer, k_features=5)
        lime_time = time.time() - start_lime

        print(f"\n[Classical LIME] time = {lime_time:.3f} sec")
        for word, weight in explanation_lime:
            print(f"   {word}: {weight:.3f}")

        #--- Q-LIME Pi
        bow = vectorizer.transform([text_sample])
        bin_features = bow.toarray()[0]

        start_qlime = time.time()
        contributions_qlime = quantum_lime_explanation(bin_features, logistic_weights)
        qlime_time = time.time() - start_qlime

        print(f"\n[Q-LIME Pi] time = {qlime_time:.3f} sec")
        for i, contrib in enumerate(contributions_qlime):
            if abs(contrib) > 1e-7:
                feat_name = vectorizer.get_feature_names_out()[i]
                print(f"   Feature {feat_name} (idx={i}): {contrib:.3f}")

        #--- Compare top-5
        lime_dict = dict(explanation_lime)
        top_words_lime = sorted(lime_dict.keys(), key=lambda w: abs(lime_dict[w]), reverse=True)[:5]

        nonzero_indices = [(i, abs(contributions_qlime[i]))
                           for i in range(len(contributions_qlime))
                           if abs(contributions_qlime[i]) > 1e-7]
        top_indices_qlime = sorted(nonzero_indices, key=lambda x: x[1], reverse=True)[:5]
        top_words_qlime = [vectorizer.get_feature_names_out()[idx2] for (idx2, val) in top_indices_qlime]

        overlap = set(top_words_lime).intersection(set(top_words_qlime))
        print(f"\nTop-5 words (Classical LIME): {top_words_lime}")
        print(f"Top-5 words (Q-LIME Pi):       {top_words_qlime}")
        print(f"Overlap in top features: {overlap}")
        print("-------------------------------------")

##############################################################################
# PART 4B: BENCHMARK TEST (for multiple max_features, etc.)
##############################################################################

def benchmark_test():
    """
    Systematically vary max_features, etc., measure:
      - test accuracy
      - LIME explanation time
      - QLIME Pi explanation time
      - overlap in top-5 features
    We'll just do a small loop for demonstration.
    """

    # Parameter combos to test
    max_features_list = [5, 10, 15]
    results = []

    for mf in max_features_list:
        # Load data
        X_train, X_test, y_train, y_test, vectorizer = load_imdb_subset(
            num_samples=500,
            min_df=1,
            max_features=mf
        )
        clf = train_logistic_classifier(X_train, y_train, vectorizer)

        X_test_bow = vectorizer.transform(X_test)
        acc = accuracy_score(y_test, clf.predict(X_test_bow))

        logistic_weights = clf.coef_[0]

        # We'll pick 10 random samples to measure average LIME & QLIME times, overlap
        sample_indices = np.random.choice(len(X_test), size=10, replace=False)

        lime_times = []
        qlime_times = []
        overlaps = []

        for idx in sample_indices:
            text_sample = X_test[idx]

            #--- LIME
            start_lime = time.time()
            explanation_lime = run_classical_lime(text_sample, clf, vectorizer, k_features=5)
            lime_times.append(time.time() - start_lime)

            #--- QLIME
            bow = vectorizer.transform([text_sample]).toarray()[0]
            start_qlime = time.time()
            contributions_qlime = quantum_lime_explanation(bow, logistic_weights)
            qlime_times.append(time.time() - start_qlime)

            # Overlap
            lime_dict = dict(explanation_lime)
            top_words_lime = sorted(lime_dict.keys(), key=lambda w: abs(lime_dict[w]), reverse=True)[:5]
            nonzero_indices = [(i, abs(contributions_qlime[i]))
                               for i in range(len(contributions_qlime))
                               if abs(contributions_qlime[i]) > 1e-7]
            top_indices_qlime = sorted(nonzero_indices, key=lambda x: x[1], reverse=True)[:5]
            top_words_qlime = [vectorizer.get_feature_names_out()[idx2] for (idx2, val) in top_indices_qlime]

            overlap = set(top_words_lime).intersection(set(top_words_qlime))
            overlaps.append(len(overlap))

        results.append({
            "max_features": mf,
            "test_accuracy": round(acc, 3),
            "lime_time_avg": round(np.mean(lime_times), 3),
            "qlime_time_avg": round(np.mean(qlime_times), 3),
            "overlap_avg": round(np.mean(overlaps), 2)
        })

    # Print results
    print("\n=== BENCHMARK RESULTS (Flip Only 1->0) ===")
    print("max_feats | Acc  | LIME_time | QLIME_time | Overlap")
    for r in results:
        print(f"{r['max_features']:9d} | "
              f"{r['test_accuracy']:.3f} | "
              f"{r['lime_time_avg']:.3f}   | "
              f"{r['qlime_time_avg']:.3f}      | "
              f"{r['overlap_avg']:.2f}")

##############################################################################
# MAIN
##############################################################################

if __name__ == "__main__":
    # 1) Run the example approach (5 random samples)
    print("RUN EXAMPLE DEMO:")
    run_example()

    # 2) Run a small benchmark test for different max_features
    print("\n\nRUN BENCHMARK TEST:")
    benchmark_test()

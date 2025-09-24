
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# ==============================
# Load Toy Dataset
# ==============================
toy_data = {
    "peptide": ["ACDE", "FGHI", "KLMN", "QRST", "VWYA", "ACDF", "ACDY", "QRSY"],
    "label":   [1, 0, 1, 0, 1, 1, 0, 0]
}
df = pd.DataFrame(toy_data)

# ==============================
# Encode Peptides (One-Hot)
# ==============================
peptides = df["peptide"].tolist()
labels = df["label"].values

enc = OneHotEncoder(handle_unknown="ignore")
X = enc.fit_transform(np.array(list(peptides)).reshape(-1,1)).toarray().reshape(len(peptides), -1)

# Train baseline logistic regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)

# ==============================
# Candidate Proposal Function
# ==============================
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def propose_candidates(peptide, mask_index=1, top_k=3):
    """
    Mask one amino acid and propose candidates.
    In this simplified demo, candidates are chosen by frequency among binders.
    """
    binders = df[df["label"] == 1]["peptide"]
    freqs = {aa:0 for aa in amino_acids}
    for pep in binders:
        if len(pep) > mask_index:
            freqs[pep[mask_index]] += 1
    top_residues = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:top_k]

    candidates = []
    for aa, _ in top_residues:
        new_pep = peptide[:mask_index] + aa + peptide[mask_index+1:]
        candidates.append(new_pep)
    return candidates

# ==============================
# Rescoring Function
# ==============================
def score_peptides(peptides):
    X_new = enc.transform(np.array(list(peptides)).reshape(-1,1)).toarray().reshape(len(peptides), -1)
    scores = clf.predict_proba(X_new)[:,1]
    return dict(zip(peptides, scores))

# ==============================
# Run Optimization on High-Affinity Peptides
# ==============================
if __name__ == "__main__":
    for pep in df[df["label"]==1]["peptide"]:
        print(f"Original: {pep}")
        candidates = propose_candidates(pep, mask_index=len(pep)//2)
        scores = score_peptides(candidates)
        for cand, score in scores.items():
            print(f"  Candidate: {cand}, Score: {score:.3f}")

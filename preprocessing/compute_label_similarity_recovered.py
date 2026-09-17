
from pathlib import Path
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

labels = {
    "Obligations": "Provisions regarding obliging each party to their responsibilities, allocated risks, rules and laws.",
    "Penalties": "Provisions regarding compensation damages made to a party or granting additional rights to the party in case of the other party's failure to perform the contract.",
    "Rights entitlement": "Provisions regarding endowing each party's rights for the purpose of protecting their assets.",
    "Behavior Restrictions": "Provisions regarding constraining each party from doing certain things.",
    "Adjudication": "Provisions regarding arbitrating disputes between the parties.",
    "Interpretations": "Provisions regarding clarifying the meanings of the terms for consensus understanding between the parties.",
    "Task descriptions": "Provisions regarding describing each party's tasks for completing the work.",
    "Communication": "Provisions regarding information exchange and sharing between the parties.",
    "Positional power": "Provisions regarding authorizing position power to each party during the process of the work.",
    "Environmental changes": "Provisions regarding dealing with natural or institutional changes in the future.",
    "Task changes": "Provisions regarding dealing with the change of work scope, standard and sequence in the future."
}

label_names = list(labels.keys())

label_texts = [
    f"{label}: {definition}"
    for label, definition in labels.items()
]

parser = argparse.ArgumentParser(description="Recovered label-definition embedding workflow; original model revision was not recorded.")
parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reproduced/semantic_similarity")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
model = SentenceTransformer(args.model)

embeddings = model.encode(
    label_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

similarity_matrix = cosine_similarity(embeddings)

sim_df = pd.DataFrame(
    similarity_matrix,
    index=label_names,
    columns=label_names
)

print(sim_df.round(3))

sim_df.to_excel(args.output_dir / "label_semantic_similarity_matrix.xlsx")
sim_df.to_csv(args.output_dir / "label_semantic_similarity_matrix.csv", encoding="utf-8-sig")

pairs = []

for i in range(len(label_names)):
    for j in range(i + 1, len(label_names)):
        pairs.append({
            "label_1": label_names[i],
            "label_2": label_names[j],
            "cosine_similarity": similarity_matrix[i, j]
        })

pair_df = pd.DataFrame(pairs).sort_values(
    by="cosine_similarity",
    ascending=False
).reset_index(drop=True)

print(pair_df.round(3))

pair_df.to_excel(args.output_dir / "label_semantic_similarity_pairs_ranked.xlsx", index=False)
pair_df.to_csv(args.output_dir / "label_semantic_similarity_pairs_ranked.csv", index=False, encoding="utf-8-sig")

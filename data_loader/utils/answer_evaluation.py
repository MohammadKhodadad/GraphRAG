import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download("punkt")

# Function to calculate ROUGE scores
def compute_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure,
    }

# Function to calculate BLEU score
def compute_bleu(reference, prediction):
    reference_tokens = nltk.word_tokenize(reference)
    prediction_tokens = nltk.word_tokenize(prediction)
    return sentence_bleu([reference_tokens], prediction_tokens)

# Function to calculate BERTScore
def compute_bert_score(reference, prediction):
    P, R, F1 = score([prediction], [reference], lang="en", verbose=False)
    return {
        "BERTScore Precision": P.mean().item(),
        "BERTScore Recall": R.mean().item(),
        "BERTScore F1": F1.mean().item(),
    }

# Function to calculate cosine similarity using Sentence-BERT
def compute_cosine_similarity(reference, prediction):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and encode
    ref_tokens = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
    pred_tokens = tokenizer(prediction, return_tensors="pt", padding=True, truncation=True)

    # Generate embeddings
    ref_embedding = model(**ref_tokens).last_hidden_state.mean(dim=1).detach().numpy()
    pred_embedding = model(**pred_tokens).last_hidden_state.mean(dim=1).detach().numpy()

    # Compute cosine similarity
    similarity = cosine_similarity(ref_embedding, pred_embedding)
    return similarity[0][0]

# Main evaluation function
def evaluate_similarity(reference, prediction):
    results = {}

    # Compute ROUGE
    rouge_scores = compute_rouge(reference, prediction)
    results.update(rouge_scores)

    # Compute BLEU
    bleu_score = compute_bleu(reference, prediction)
    results["BLEU"] = bleu_score

    # Compute BERTScore
    bert_scores = compute_bert_score(reference, prediction)
    results.update(bert_scores)

    # Compute Cosine Similarity
    cosine_sim = compute_cosine_similarity(reference, prediction)
    results["Cosine Similarity"] = cosine_sim

    return results

# Example usage
if __name__ == "__main__":
    reference_text = "The Eiffel Tower is one of the most famous landmarks in the world, located in Paris, France."
    prediction_text = "The Eiffel Tower, located in Paris, is a well-known global landmark."

    metrics = evaluate_similarity(reference_text, prediction_text)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

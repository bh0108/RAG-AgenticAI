import pdfplumber
import nltk
from transformers import pipeline
nltk.download()
# -----------------------------
# 1. Extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

pdf_path = "google_terms.pdf"   # Place your PDF in the same folder
raw_text = extract_text_from_pdf(pdf_path)

# -----------------------------
# 2. Preview extracted text
# -----------------------------
print("\n--- Preview of Extracted Text ---\n")
print(raw_text[:1000])   # Only preview first 1000 chars

# -----------------------------
# 3. Summarize the document
# -----------------------------
print("\n--- Summary ---\n")

from transformers import pipeline

summarizer = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

summary = summarizer(
    f"Summarize the following text:\n\n{raw_text[:4000]}",
    max_length=200,
    min_length=80,
    do_sample=False
)

print(summary[0]['generated_text'])

# -----------------------------
# 4. Split into sentences & passages
# -----------------------------
print("\n--- Splitting into Sentences & Passages ---\n")

#nltk.download('punkt')

sentences = nltk.sent_tokenize(raw_text)
passages = [" ".join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]

print(f"Total sentences: {len(sentences)}")
print(f"Total passages: {len(passages)}")
print("\nSample Passage:\n", passages[0])

# -----------------------------
# 5. Generate questions using LLM
# -----------------------------
print("\n--- Generating Questions ---\n")

qg = pipeline("text-generation", model="google/flan-t5-base")

def generate_questions(passage):
    prompt = f"Generate 3 questions based on the following text:\n{passage}"
    output = qg(prompt, max_length=128)
    return output[0]['generated_text']

sample_questions = generate_questions(passages[0])
print(sample_questions)

# -----------------------------
# 6. Answer questions using QA model
# -----------------------------
print("\n--- Answering Questions ---\n")

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

questions_list = [q for q in sample_questions.split("\n") if q.strip()]

for q in questions_list:
    answer = qa(question=q, context=passages[0])
    print("Q:", q)
    print("A:", answer['answer'])
    print()
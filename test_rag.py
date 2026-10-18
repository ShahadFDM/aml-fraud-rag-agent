import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load the policy document
policy_file = "aml_compliance_policy.txt"
if not os.path.exists(policy_file):
    raise FileNotFoundError(f"Missing {policy_file}. Ensure it exists in your working directory.")

loader = TextLoader(policy_file)
docs = loader.load()

# 2. Rule-level splitting
# Chunk size 600 ensures each of the 4 rules forms exactly one document chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=0,
    separators=["\n\nRule ", "\n\n"]
)
chunks = text_splitter.split_documents(docs)
print(f"Total policy chunks indexed: {len(chunks)}")

# 3. Initialize local embeddings and create the FAISS index
print("Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save index locally for production reuse in app.py
vectorstore.save_local("faiss_aml_index")
print("Saved FAISS vector store to 'faiss_aml_index'.")

# 4. Configure top-2 retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 5. Define a 12-case benchmark to test retrieval accuracy
# Each case supplies all policy fields and its ground-truth expected rule(s)
eval_benchmark = [
    {
        "id": 1,
        "desc": "High amount TRANSFER over 200k",
        "tx": "Type: TRANSFER | Amount: 350000.0 | Old Balance Org: 500000.0 | New Balance Orig: 150000.0 | Dest ID: C998877 | Old Balance Dest: 100000.0 | New Balance Dest: 450000.0",
        "expected": ["Rule 101"]
    },
    {
        "id": 2,
        "desc": "High amount CASH_OUT over 200k",
        "tx": "Type: CASH_OUT | Amount: 220000.0 | Old Balance Org: 300000.0 | New Balance Orig: 80000.0 | Dest ID: C112233 | Old Balance Dest: 50000.0 | New Balance Dest: 270000.0",
        "expected": ["Rule 101"]
    },
    {
        "id": 3,
        "desc": "Account takeover depleting origin balance to 0",
        "tx": "Type: TRANSFER | Amount: 45000.0 | Old Balance Org: 45000.0 | New Balance Orig: 0.0 | Dest ID: C554433 | Old Balance Dest: 12000.0 | New Balance Dest: 57000.0",
        "expected": ["Rule 102"]
    },
    {
        "id": 4,
        "desc": "CASH_OUT draining entire account balance to zero",
        "tx": "Type: CASH_OUT | Amount: 95000.0 | Old Balance Org: 95000.0 | New Balance Orig: 0.0 | Dest ID: C223344 | Old Balance Dest: 4000.0 | New Balance Dest: 99000.0",
        "expected": ["Rule 102"]
    },
    {
        "id": 5,
        "desc": "TRANSFER to recipient mule account with initial zero balance",
        "tx": "Type: TRANSFER | Amount: 75000.0 | Old Balance Org: 150000.0 | New Balance Orig: 75000.0 | Dest ID: C445566 | Old Balance Dest: 0.0 | New Balance Dest: 75000.0",
        "expected": ["Rule 103"]
    },
    {
        "id": 6,
        "desc": "Large transfer matching both Rule 101 (>200k) and Rule 103 (mule dest 0)",
        "tx": "Type: TRANSFER | Amount: 320000.0 | Old Balance Org: 400000.0 | New Balance Orig: 80000.0 | Dest ID: C778899 | Old Balance Dest: 0.0 | New Balance Dest: 320000.0",
        "expected": ["Rule 101", "Rule 103"]
    },
    {
        "id": 7,
        "desc": "Large PAYMENT over 100k to merchant account starting with M",
        "tx": "Type: PAYMENT | Amount: 140000.0 | Old Balance Org: 200000.0 | New Balance Orig: 60000.0 | Dest ID: M987654321 | Old Balance Dest: 0.0 | New Balance Dest: 0.0",
        "expected": ["Rule 104"]
    },
    {
        "id": 8,
        "desc": "Drain to zero (>200k) matching Rule 101 and Rule 102",
        "tx": "Type: TRANSFER | Amount: 260000.0 | Old Balance Org: 260000.0 | New Balance Orig: 0.0 | Dest ID: C123987 | Old Balance Dest: 50000.0 | New Balance Dest: 310000.0",
        "expected": ["Rule 101", "Rule 102"]
    },
    {
        "id": 9,
        "desc": "High-value merchant payment exceeding 100k threshold",
        "tx": "Type: PAYMENT | Amount: 150000.0 | Old Balance Org: 5000.0 | New Balance Orig: 3500.0 | Dest ID: M1234567 | Old Balance Dest: 0.0 | New Balance Dest: 0.0",
        "expected": ["Rule 104"]
    },
    {
        "id": 10,
        "desc": "TRANSFER over 50k to an empty recipient account",
        "tx": "Type: TRANSFER | Amount: 80000.0 | Old Balance Org: 120000.0 | New Balance Orig: 40000.0 | Dest ID: C556677 | Old Balance Dest: 0.0 | New Balance Dest: 80000.0",
        "expected": ["Rule 103"]
    },
    {
        "id": 11,
        "desc": "CASH_OUT emptying account to 0 with amount over 200k",
        "tx": "Type: CASH_OUT | Amount: 400000.0 | Old Balance Org: 400000.0 | New Balance Orig: 0.0 | Dest ID: C887766 | Old Balance Dest: 10000.0 | New Balance Dest: 410000.0",
        "expected": ["Rule 101", "Rule 102"]
    },
    {
        "id": 12,
        "desc": "Merchant payment exceeding threshold",
        "tx": "Type: PAYMENT | Amount: 185000.0 | Old Balance Org: 300000.0 | New Balance Orig: 115000.0 | Dest ID: M554433221 | Old Balance Dest: 0.0 | New Balance Dest: 0.0",
        "expected": ["Rule 104"]
    }
]

# 6. Execute retrieval and evaluate accuracy
correct_count = 0
total_cases = len(eval_benchmark)

print("\n--- RUNNING BENCHMARK EVALUATION (k=2) ---\n")

for test in eval_benchmark:
    retrieved_docs = retriever.invoke(test["tx"])
    retrieved_text = " ".join([d.page_content for d in retrieved_docs])
    
    # Check if every expected rule appears in the top-2 retrieved documents
    is_correct = all(rule in retrieved_text for rule in test["expected"])
    if is_correct:
        correct_count += 1
        status = "PASSED"
    else:
        status = "FAILED"
        
    print(f"Case {test['id']:02d} [{status}] - Expected: {test['expected']}")
    for rank, doc in enumerate(retrieved_docs, start=1):
        rule_header = doc.page_content.strip().split("\n")[0]
        print(f"   Rank {rank}: {rule_header}")
    print()

retrieval_acc = (correct_count / total_cases) * 100
print("=" * 60)
print(f"EVALUATION RESULT: {correct_count}/{total_cases} cases correctly retrieved.")
print(f"RAG Top-2 Retrieval Accuracy: {retrieval_acc:.1f}%")
print("=" * 60)

# Note:
# A negative/non-triggering transaction should be evaluated separately from
# retrieval accuracy because a top-k vector retriever will always return k
# documents, even when none of the rules actually applies.
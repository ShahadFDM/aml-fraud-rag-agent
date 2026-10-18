import json
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


PROJECT_DIR = Path(__file__).resolve().parent
POLICY_FILE = PROJECT_DIR / "aml_compliance_policy.txt"
MODEL_FILE = PROJECT_DIR / "fraud_model.joblib"
FAISS_DIR = PROJECT_DIR / "faiss_aml_index"


def load_vectorstore():
    """Load the saved FAISS index or build it from the policy document."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if FAISS_DIR.exists():
        return FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if not POLICY_FILE.exists():
        raise FileNotFoundError(
            f"Missing policy file: {POLICY_FILE}"
        )

    print("FAISS index not found. Building it from the AML policy...")
    docs = TextLoader(str(POLICY_FILE), encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=0,
        separators=["\n\nRule ", "\n\n"],
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_DIR))
    print("FAISS index created successfully.")

    return vectorstore


def get_float(prompt):
    """Read a numeric value from the user."""
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("Please enter a valid number.")


def get_int(prompt):
    """Read an integer value from the user."""
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("Please enter a whole number.")


def get_nonempty(prompt):
    """Read a non-empty text value."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Please enter a value.")


def main():
    load_dotenv(PROJECT_DIR / ".env")

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if not groq_api_key:
        raise ValueError(
            "Missing GROQ_API_KEY. Create a local .env file using .env.example."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_FILE}. "
            "Run 2_model_training.ipynb first."
        )

    print("=" * 70)
    print("AML FRAUD DETECTION & POLICY-GROUNDED ALERT DEMO")
    print("=" * 70)

    # Load the trained model artifact.
    model_artifact = joblib.load(MODEL_FILE)
    model = model_artifact["pipeline"]
    threshold = float(model_artifact["threshold"])
    features = model_artifact["features"]

    # Collect a new transaction from the user.
    step = get_int("Step (hour): ")
    tx_type = get_nonempty(
        "Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER): "
    ).upper()
    amount = get_float("Amount: ")

    oldbalance_org = get_float("Old balance of origin: ")
    newbalance_orig = get_float("New balance of origin: ")
    name_dest = get_nonempty("Destination ID: ")

    # PaySim does not track balances for merchant IDs that begin with M.
    # Its dataset represents those unavailable balances as 0.0, so fill them
    # automatically and avoid asking the user to enter values that are unknown.
    is_merchant = name_dest.upper().startswith("M")
    if is_merchant:
        oldbalance_dest = 0.0
        newbalance_dest = 0.0
        print(
            "Merchant destination detected (ID starts with 'M'). "
            "Destination balances are not tracked in PaySim; using 0.0 placeholders."
        )
    else:
        oldbalance_dest = get_float("Old balance of destination: ")
        newbalance_dest = get_float("New balance of destination: ")

    transaction = {
        "step": step,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "nameDest": name_dest,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
    }

    row_df = pd.DataFrame([transaction])

    # The fraud model deliberately uses only the non-leaking features.
    model_input = row_df[features]
    risk_score = float(model.predict_proba(model_input)[0, 1])
    risk_level = "HIGH" if risk_score >= threshold else "LOW"

    print("\n" + "-" * 70)
    print("FRAUD MODEL RESULT")
    print("-" * 70)
    print(f"Fraud Risk Score: {risk_score:.3f}")
    print(f"Risk Level:       {risk_level}")
    print(f"Decision Threshold: {threshold:.3f}")

    # The RAG layer uses policy-relevant transaction fields.
    tx_details = (
        f"Type: {tx_type} | "
        f"Amount: {amount} | "
        f"Old Balance Org: {oldbalance_org} | "
        f"New Balance Orig: {newbalance_orig} | "
        f"Dest ID: {name_dest} | "
        f"Old Balance Dest: {oldbalance_dest}"
        + (" (merchant balance not tracked; 0.0 is a placeholder)" if is_merchant else "")
        + f" | New Balance Dest: {newbalance_dest}"
        + (" (merchant balance not tracked; 0.0 is a placeholder)" if is_merchant else "")
    )

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(tx_details)

    policy_rules = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    if not policy_rules:
        policy_rules = "No specific corporate policy rule was retrieved."

    print("\n" + "-" * 70)
    print("RETRIEVED POLICY")
    print("-" * 70)
    for rank, doc in enumerate(retrieved_docs, start=1):
        first_line = doc.page_content.strip().splitlines()[0]
        print(f"Rank {rank}: {first_line}")

    prompt_template = PromptTemplate(
        input_variables=[
            "risk_score",
            "risk_level",
            "transaction_details",
            "compliance_rules",
        ],
        template=(
            "You are an AML alert-writing assistant.\n\n"
            "The fraud model has already calculated the following result. "
            "Treat it as a fixed fact and do not change it:\n"
            "Fraud Risk Score: {risk_score}\n"
            "Risk Level: {risk_level}\n\n"
            "Retrieved corporate AML policy rules:\n"
            "{compliance_rules}\n\n"
            "Transaction details:\n"
            "{transaction_details}\n\n"
            "Use ONLY the supplied transaction information and policy rules.\n"
            "Do not invent AML regulations.\n"
            "Do not cite external laws or regulations.\n"
            "If no retrieved rule applies, return an empty policy_rules list "
            "and state that no matching corporate policy was found.\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            "{{\n"
            '  "policy_rules": ["Rule 101"],\n'
            '  "alert_summary": "Two concise professional sentences.",\n'
            '  "recommended_action": "Manual review"\n'
            "}}"
        ),
    )

    final_prompt = prompt_template.format(
        risk_score=f"{risk_score:.3f}",
        risk_level=risk_level,
        transaction_details=tx_details,
        compliance_rules=policy_rules,
    )

    print("\n" + "-" * 70)
    print(f"CONNECTING TO GROQ: {groq_model}")
    print("-" * 70)

    llm = ChatGroq(
        model_name=groq_model,
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}},
        api_key=groq_api_key,
    )

    response = llm.invoke(final_prompt)

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        print("The LLM did not return valid JSON.")
        print(response.content)
        return

    print("\n" + "=" * 70)
    print("AML ALERT")
    print("=" * 70)
    print(
        "Policy Rules:   "
        + ", ".join(result.get("policy_rules", []))
    )
    print(
        "Alert Summary:  "
        + str(result.get("alert_summary", ""))
    )
    print(
        "Action:         "
        + str(result.get("recommended_action", ""))
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# 1. Load Environment Variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY. Please add it to your .env file.")


# 2. Initialize the Groq LLM
# We use the requested openai/gpt-oss-20b model and force JSON output
print("Connecting to Groq LLM (openai/gpt-oss-20b)...")
llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0.1,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# 3. Load the existing FAISS Vector Database
print("Loading FAISS vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_aml_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. Load the transaction data
print("Loading transaction data...")
df = pd.read_csv("sampled_transactions.csv")
fraud_cases = df[df['isFraud'] == 1].head(3) # Testing on 3 cases for speed

# 5. Define the Structured Prompt Template
prompt_template = PromptTemplate(
    input_variables=["transaction_details", "compliance_rule"],
    template=(
        "You are an expert Anti-Money Laundering (AML) compliance officer. "
        "Evaluate the following transaction against the provided Corporate AML Policy rules.\n\n"
        "Corporate AML Policy Rules:\n{compliance_rule}\n\n"
        "Transaction Details:\n{transaction_details}\n\n"
        "INSTRUCTIONS:\n"
        "- Use ONLY the supplied transaction data and retrieved corporate policy.\n"
        "- Do not invent AML rules or cite external regulations.\n"
        "- If no relevant policy applies, state that no matching corporate policy was found.\n"
        "- Respond STRICTLY in the following JSON format. Do not include any markdown formatting or outside text.\n"
        "{{\n"
        '  "risk_level": "HIGH or LOW",\n'
        '  "policy_rules": ["Rule ID(s) violated, or None"],\n'
        '  "alert_summary": "Short, professional 2-sentence explanation of the violation.",\n'
        '  "recommended_action": "e.g., Manual review, Enhanced Due Diligence, etc."\n'
        "}}\n"
    )
)

# 6. Execute the Pipeline
print("\n--- GENERATING STRUCTURED AML ALERTS ---\n")

for index, row in fraud_cases.iterrows():
    # A. Format the transaction context
    tx_details = f"Type: {row['type']} | Amount: {row['amount']} | Old Balance Org: {row['oldbalanceOrg']} | New Balance Orig: {row['newbalanceOrig']} | Dest ID: {row['nameDest']} | Old Balance Dest: {row['oldbalanceDest']} | New Balance Dest: {row['newbalanceDest']}"
    
    # B. Retrieve top 2 rules from FAISS
    retrieved_docs = retriever.invoke(tx_details)
    policy_rules = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # C. Format the prompt and invoke Groq
    final_prompt = prompt_template.format(
        transaction_details=tx_details, 
        compliance_rule=policy_rules
    )
    
    response = llm.invoke(final_prompt)
    
    # D. Parse and display the JSON output
    try:
        alert_data = json.loads(response.content)
        print(f"Transaction ID #{index}: {row['type']} of ${row['amount']}")
        print(f"Risk Level:         {alert_data.get('risk_level')}")
        print(f"Rules Flagged:      {', '.join(alert_data.get('policy_rules', []))}")
        print(f"Alert Summary:      {alert_data.get('alert_summary')}")
        print(f"Recommended Action: {alert_data.get('recommended_action')}")
        print("-" * 60)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for Transaction ID #{index}. Raw output:\n{response.content}")
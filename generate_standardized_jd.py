import os
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging
import argparse
import re
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# --- Configuration ---
CHROMA_PERSIST_DIR = "chroma_db_basic_chunks"
CHROMA_COLLECTION_NAME = "jd_basic_chunks"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OUTPUT_DIR = "outputs"

if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    exit()

def get_user_comprehensive_query():
    print("\n" + "="*80)
    print("🎯 JOB DESCRIPTION GENERATION ASSISTANT")
    print("="*80)
    print("Please provide ALL job requirements in a single, comprehensive prompt.\n")
    comprehensive_query = input("📝 Enter your comprehensive job requirements: ").strip()
    while not comprehensive_query:
        comprehensive_query = input("❗ Please enter some job requirements: ").strip()
    print("\n" + "-"*80)
    print(f"   • {comprehensive_query}")
    print("-"*80)
    return comprehensive_query

# def validate_comprehensive_query(query: str) -> str:
    try:
        validation_prompt = f"""
        You are a professional HR assistant reviewing a director's input for generating a job description.

        The following is the DIRECTOR'S COMPREHENSIVE JOB REQUIREMENTS:
        >>> {query}

        1. If the input is too vague, empty, or nonsensical (e.g., just one unrelated word or random characters),
           respond with a polite and professional message asking for clearer job requirements.

        2. The input might be in short form like SE or se (don't focus on the case) for Software Engineer, ds for Data Scientist, etc.
           **Make sure for any short form input which is choosely used, expand it to its full form.**

        3. If the input is meaningful and usable as job requirements, return only the original input unmodified.

        Output:
        - A valid query string for JD generation OR
        - A professionally phrased response to the director, including the original input as context. 
           NOTE: just the message without any thinking or explanation.
        """

        validation_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )

        parser = StrOutputParser()
        validate_chain = RunnableLambda(lambda x: validation_llm.invoke(x).content) | parser

        result = validate_chain.invoke(validation_prompt.strip())
        return result.strip()
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return "[RESPONSE TO DIRECTOR] We encountered an issue validating the input. Please try again."

def retrieve_relevant_chunks(query_text, top_k=8):
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        query_embedding = embeddings_model.embed_query(query_text)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results['documents'][0] if results and results.get('documents') and results['documents'][0] else []
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        return []

def generate_standardized_jd(comprehensive_query, relevant_chunks):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )
        context = "\n\n---\n\n".join(relevant_chunks)

        template = """
        You are a professional HR content writer and recruitment specialist, acting as an assistant to a director.
        Your task is to create a comprehensive, professional job description. You will base this solely on the director's single, comprehensive query and the provided reference content from similar positions.

        **PRIMARY INSTRUCTION**: Generate the job description based *only* on the information provided in the 'DIRECTOR'S COMPREHENSIVE JOB REQUIREMENTS' section below. If certain details are not explicitly mentioned in the requirements, infer them logically based on the stated job title or role context, and use the 'REFERENCE JOB DESCRIPTIONS' for guidance.

        DIRECTOR'S COMPREHENSIVE JOB REQUIREMENTS:
        {comprehensive_query}

        REFERENCE JOB DESCRIPTIONS:
        {context}

        Strictly adhere to the following markdown format:

        # Job Description: [INFER JOB TITLE HERE]

        ## 1. About Pangea
        [Write a compelling 2-3 sentence paragraph about the company...]

        ## 2. About the Role and Key Responsibilities
        [Detailed paragraph and 6-8 bullet points]

        ## 3. Must Have Skills & Qualifications
        [6-8 essential bullet points]

        ## 4. Nice to Have Skills
        [4-6 additional bullet points]

        ---
        Ensure all details are inferred clearly and the tone is aligned with reference examples.
        **NOTE: No thinkg or explanation, just the job description in the mentioned structure.**
        """

        prompt = ChatPromptTemplate.from_template(template)
        formatted_prompt = prompt.format(
            comprehensive_query=comprehensive_query,
            context=context
        )

        print("\n🔄 Generating job description...")
        response = llm.invoke(formatted_prompt)
        jd_content = response.content

        title_match = re.search(r'#\s*Job Description:\s*(.+?)\n', jd_content)
        job_title = title_match.group(1).strip() if title_match else "untitled_job"
        job_title = re.sub(r'[^\w\s-]', '', job_title)[:50].strip()

        return jd_content, job_title
    except Exception as e:
        logging.error(f"Error generating job description: {e}")
        return f"Error generating job description: {e}", "error_jd"

def save_job_description(job_description_content, job_title_for_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = job_title_for_filename.replace(' ', '_').replace('/', '_').lower()
    output_filename = f"{safe_title}_{timestamp}_jd.md"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(job_description_content)
        print(f"\n💾 Job description saved to: {output_filepath}")
        return output_filepath
    except Exception as e:
        logging.error(f"Error saving to file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Interactive Job Description Generation Assistant')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--query', type=str)
    parser.add_argument('--top_k', type=int, default=8)
    args = parser.parse_args()

    comprehensive_query = args.query if args.batch else get_user_comprehensive_query()

    # validated_input = validate_comprehensive_query(comprehensive_query)
    # if validated_input.startswith("[RESPONSE TO DIRECTOR]"):
    #     print("\n✉️ Response to Director:")
    #     print(validated_input)
    #     return
    # else:
    #     comprehensive_query = validated_input

    print(f"\n🔍 Searching for relevant content...")
    relevant_chunks = retrieve_relevant_chunks(comprehensive_query, args.top_k)

    if not relevant_chunks:
        print("❌ No relevant content found. Ensure your vector DB is populated.")
        return

    print(f"✅ Found {len(relevant_chunks)} relevant content pieces.")

    jd_content, job_title_for_filename = generate_standardized_jd(comprehensive_query, relevant_chunks)
    output_file = save_job_description(jd_content, job_title_for_filename)

    print("\n" + "="*80)
    print("📄 GENERATED JOB DESCRIPTION")
    print("="*80)
    print(jd_content)
    print("="*80)

    if output_file:
        print(f"\n✅ Job description generation complete!")
        print(f"📁 File saved: {output_file}")

    logging.info("Process complete.")

if __name__ == "__main__":
    main()

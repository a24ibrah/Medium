"""
HR Knowledge Graph Builder
=========================
This script creates a Neo4j knowledge graph for HR data with:
- Employee/department/skill relationships
- Policy document chunking and local embeddings
- Hybrid search (structured + semantic)

Designed to run offline using HuggingFace embeddings.

SETUP:
1. Install Neo4j Desktop (https://neo4j.com/download/)
2. Create a local Neo4j DB (default: bolt://localhost:7687, user=neo4j, password=password)
3. Install dependencies: pip install -r requirements.txt
"""

# === IMPORTS ===
from langchain.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # Local embeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
import pandas as pd
import os

# === CONFIGURATION ===
# Neo4j Configuration (local default)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change to your Neo4j password

# Embeddings Model (runs locally)
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"  # Lightweight local model

# File Paths
EMPLOYEES_CSV = "data/employees.csv"
SKILLS_CSV = "data/skills.csv"
POLICY_PDF = "data/employee_handbook.pdf"  # Sample text file for demo


# === INITIALIZE NEO4J CONNECTION ===
def init_neo4j():
    """Initialize connection to Neo4j database"""
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        print("✅ Neo4j connection established")
        return graph
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        raise


# === DATA INGESTION FUNCTIONS ===
def load_hr_data(graph):
    """
    Load HR data into Neo4j with:
    - Employees
    - Departments
    - Skills
    - Employee-skill relationships
    """
    # Clear existing data (for demo)
    graph.run("MATCH (n) DETACH DELETE n")

    # Create constraints for uniqueness
    graph.run("CREATE CONSTRAINT unique_employee IF NOT EXISTS FOR (e:Employee) REQUIRE e.id IS UNIQUE")
    graph.run("CREATE CONSTRAINT unique_dept IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE")

    # Load employees and departments
    employees_df = pd.read_csv(EMPLOYEES_CSV)
    for _, row in employees_df.iterrows():
        graph.run("""
        MERGE (e:Employee {id: $id})
        SET e.name = $name, 
            e.role = $role,
            e.hire_date = date($hire_date),
            e.salary = toInteger($salary)
        MERGE (d:Department {name: $dept})
        SET d.location = $location
        MERGE (e)-[:WORKS_IN]->(d)
        """, {
            'id': row['id'],
            'name': row['name'],
            'role': row['role'],
            'hire_date': row['hire_date'],
            'salary': row['salary'],
            'dept': row['department'],
            'location': row.get('location', 'Unknown')  # Default if missing
        })

    # Load skills and relationships
    skills_df = pd.read_csv(SKILLS_CSV)
    for _, row in skills_df.iterrows():
        graph.run("""
        MATCH (e:Employee {id: $emp_id})
        MERGE (s:Skill {name: $skill})
        SET s.category = $category
        MERGE (e)-[:HAS_SKILL]->(s)
        """, {
            'emp_id': row['employee_id'],
            'skill': row['skill'],
            'category': row['category']
        })

    print("✅ HR data loaded into Neo4j")


# === DOCUMENT PROCESSING ===
def process_policy_documents(graph):
    """
    Chunk and embed policy documents using local embeddings
    """
    # Initialize local embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # Sample policy text (replace with PDF/text extraction in production)
    policy_text = """
    Remote Work Policy:
    Employees may work remotely up to 3 days per week with manager approval.

    Leave Policy:
    Annual leave accrues at 15 days/year for the first 5 years of service.
    """

    # Chunk the document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(policy_text)

    # Create vector index
    Neo4jVector.from_texts(
        texts=chunks,
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="policy_chunks",
        node_label="PolicyChunk",
        text_node_property="text",
        embedding_node_property="embedding"
    )

    print("✅ Policy documents processed and embedded")


# === QUERY FUNCTIONS ===
def structured_query(graph, department, min_rating):
    """
    Find high-performing employees in a department (structured query)
    """
    query = """
    MATCH (e:Employee)-[:WORKS_IN]->(d:Department {name: $dept})
    MATCH (e)-[:HAS_REVIEW]->(r:Performance)
    WHERE r.rating >= $min_rating
    RETURN e.name AS name, e.role AS role, r.rating AS rating
    ORDER BY r.rating DESC
    """
    return graph.run(query, {'dept': department, 'min_rating': min_rating}).data()


def semantic_search(graph, question, k=3):
    """
    Search policy documents using semantic search
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="policy_chunks",
        text_node_property="text",
        embedding_node_property="embedding"
    )

    return vector_store.similarity_search(question, k=k)


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Initialize connection
    graph = init_neo4j()

    # Load HR data
    print("\nLoading HR data...")
    load_hr_data(graph)

    # Process policies
    print("\nProcessing policy documents...")
    process_policy_documents(graph)

    # Example queries
    print("\n=== Structured Query ===")
    print("High performers in Engineering:")
    results = structured_query(graph, "Engineering", 4.0)
    for record in results:
        print(f"- {record['name']} ({record['role']}): Rating {record['rating']}")

    print("\n=== Semantic Search ===")
    print("Remote work policy results:")
    policy_results = semantic_search(graph, "Can I work from home?")
    for i, doc in enumerate(policy_results, 1):
        print(f"{i}. {doc.page_content[:100]}...")

    print("\n✅ Tutorial completed successfully!")
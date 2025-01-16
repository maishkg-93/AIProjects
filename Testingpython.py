import streamlit as st
import boto3
from py2neo import Graph
from PyPDF2 import PdfReader
import json
import networkx as nx
import matplotlib.pyplot as plt

# AWS Bedrock Client Initialization
bedrock_client = boto3.client('bedrock', region_name='eu-west-1')

# Neo4j Initialization
neo4j_uri = "bolt://localhost:7687"  # Update with your Neo4j instance URI
neo4j_user = "neo4j"  # Update with your Neo4j username
neo4j_password = "password"  # Update with your Neo4j password
graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Function to invoke Bedrock model
def generate_response_from_bedrock(prompt):
    try:
        response = bedrock_client.invoke_model(
            modelId="eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=json.dumps({"input": prompt}),
            contentType="application/json",
            accept="application/json"
        )
        # Extract the response from the result
        response_body = response['body'].read().decode('utf-8')
        response_json = json.loads(response_body)
        return response_json.get("completion", "No response from model")
    except Exception as e:
        return f"Error invoking Bedrock model: {e}"

# Function to process PDF and create Neo4j graph
def process_pdf_and_create_graph(pdf_file):
    # Extract text from PDF
    pdf_reader = PdfReader(pdf_file)
    pdf_text = "\n".join(page.extract_text() for page in pdf_reader.pages)

    # Create a Document node
    graph.run("CREATE (doc:Document {title: $title, content: $content})",
              title=pdf_file.name, content=pdf_text)

    # Split text into words and create Word nodes
    words = pdf_text.split()
    for word in words:
        graph.run("MERGE (w:Word {name: $word})", word=word)

    # Create relationships between consecutive words
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        # Create a relationship between consecutive words
        graph.run("""
            MATCH (w1:Word {name: $word1}), (w2:Word {name: $word2})
            MERGE (w1)-[:NEXT]->(w2)
        """, word1=word1, word2=word2)

    # Visualize graph nodes and relationships
    nodes = graph.run("MATCH (n) RETURN n LIMIT 25").data()
    edges = graph.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25").data()
    return nodes, edges

# Function to create NetworkX graph from Neo4j data
def create_networkx_graph(nodes, edges):
    G = nx.DiGraph()  # Directed Graph
    for node in nodes:
        # Check if 'title' is not None before adding it to the graph
        title = node['n'].get('title', None)
        if title:
            G.add_node(title)
        else:
            # Handle the case where the title is missing or None
            st.warning("Warning: Node with missing or None title found and skipped.")
    
    for edge in edges:
        # Check if the source and target are not None before adding an edge
        source = edge['n'].get('title', None)
        target = edge['m'].get('title', None)
        if source and target:
            G.add_edge(source, target)
        else:
            # Handle missing or None values in relationships
            st.warning("Warning: Edge with missing or None source/target found and skipped.")
    
    return G

# Streamlit UI
st.title("Document Analysis with Bedrock and Neo4j")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        nodes, edges = process_pdf_and_create_graph(uploaded_file)
        st.success("PDF processed and graph created in Neo4j!")

        # Display extracted nodes
        st.subheader("Extracted Graph Nodes")
        st.json(nodes)

        # Display extracted edges
        st.subheader("Extracted Graph Relationships")
        st.json(edges)

        # Show Graph button
        if st.button("Show Graph"):
            G = create_networkx_graph(nodes, edges)
            
            # Plot the graph using Matplotlib
            plt.figure(figsize=(10, 10))
            nx.draw(G, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
            st.pyplot(plt)

# Bedrock interaction section
st.subheader("Chat with Bedrock Model")
user_input = st.text_input("Enter your query:")
if user_input:
    with st.spinner("Generating response from Bedrock..."):
        response = generate_response_from_bedrock(user_input)
        st.success("Response received!")
        st.text_area("Claude's Response:", response)

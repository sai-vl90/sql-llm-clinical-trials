import streamlit as st
import time
import pyodbc
import pandas as pd
from sqlalchemy.engine import URL
from langchain_community.llms import AzureOpenAI
from langchain_community.utilities import SQLDatabase
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
import os
import re
from langchain_openai import AzureChatOpenAI

os.environ["AZURE_OPENAI_API_KEY"] = "884c6ad5c66a46eea52b28b352056c5c"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://amschatbot.openai.azure.com/"

st.title("🤖 Unlocking Clinical Trial Insights: Leveraging LLMs for Database Queries")

# Define a set of valid dataset columns
VALID_COLUMNS = {
    'id_info',
    'overall_status',
    'study_type',
    'eligibility_minimum_age',
    'location_countries',
    'study_first_submitted',
    'last_update_posted',
    'phase',
    'intervention_type',
    'intervention_name',
    'completion_date',
    'enrollment',
    'condition'
}

def extract_table_columns(sql_query):
    """
    Extracts all column references from a SQL query based on the dataset's valid columns.
    Supports both 'Table.Column' and 'Column' formats.
    """
    # Regular expression to capture table.column or column
    column_pattern = r'\b\w+\.(\w+)\b|\b(\w+)\b'
    matches = re.findall(column_pattern, sql_query)
    
    columns = set()
    for match in matches:
        # match is a tuple like ('column', '') or ('', 'column')
        column = match[0] if match[0] else match[1]
        if column in VALID_COLUMNS:
            columns.add(column)
    
    return list(columns)

def extract_citation_values(response, citations):
    """
    Extracts the values associated with each citation in the response.
    
    Args:
        response (str): The LLM's generated response.
        citations (list): List of citations like [Source: Table.column]
    
    Returns:
        dict: Mapping from column to list of values
    """
    citation_values = {}
    for cite in citations:
        # Extract the column name from the citation
        column = cite.split('.')[-1][:-1]  # Removes ']' at the end
        # Define a regex pattern to find the value before the citation
        # e.g., 'is "temozolomide" [Source: Table.intervention_name]'
        pattern = r'["\']([^"\']+)["\']\s*\[Source:\s*Table\.{}\]'.format(column)
        matches = re.findall(pattern, response)
        if matches:
            # Assume the last match is the relevant one
            value = matches[-1]
            if column in citation_values:
                citation_values[column].append(value)
            else:
                citation_values[column] = [value]
    return citation_values

# Function to map citation values to data rows
def map_citation_values_to_rows(citation_values, sql_result):
    """
    Maps the extracted citation values to the corresponding rows in the SQL result.
    
    Args:
        citation_values (dict): Mapping from column to list of values.
        sql_result (pd.DataFrame): The DataFrame obtained from executing the SQL query.
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing only the referenced rows.
    """
    if sql_result.empty:
        return pd.DataFrame()
    
    # Initialize a boolean series for filtering
    filter_series = pd.Series([False] * len(sql_result))
    
    for column, values in citation_values.items():
        if column in sql_result.columns:
            # Update the filter_series to include rows where column matches any of the values
            filter_series = filter_series | sql_result[column].isin(values)
    
    # Filter the DataFrame
    referenced_rows = sql_result[filter_series]
    
    return referenced_rows


def generate_sql_code(prompt):
    # check the local ODBC driver and make sure it match with the target database instance
    for driver in pyodbc.drivers():
        print(driver)
    server = 'appdbserver082.database.windows.net'
    database = 'appdb'
    username = 'sqladmin'
    password = 'Garud@123'
    driver =   'ODBC+Driver+18+for+SQL+Server'
    table = "clinical_trial_dboard"
    
    conn_string = 'DRIVER='+driver+';SERVER=tcp:'+server+',1433'+';DATABASE='+database+';UID='+username+';PWD='+password+';Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30'
    conn_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_string})
    db = SQLDatabase.from_uri(conn_url)
    
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, provide a clear and well-structured answer.
        
        - Format the answer in bullet points or a table where appropriate.
        - Ensure that the answer is precise, relevant, and concise.
        - If the data contains repeating values, avoid redundancy in your response.
        - If the question asks for a list, structure it in a list format.
        - If a calculation is involved, clearly show the calculation steps or the result.
        - If the question is out of context or cannot be answered from the database, respond with: "Irrelevant question, I cannot answer it."

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}

    Answer: """
    )
    
    llm = AzureChatOpenAI(
    openai_api_version="2023-06-01-preview",
    azure_deployment="gpt-4",
    temperature = 0.01
)
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)
    chain = write_query | execute_query
    answer = answer_prompt | llm | StrOutputParser()
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )
    sql_chain = create_sql_query_chain(llm, db=db)
    sql_code = sql_chain.invoke({"question": prompt})
    response = chain.invoke({"question": prompt})
    return sql_code, response

# Create a bigger textbox for the user to enter the prompt
prompt = st.text_area("Enter your prompt:", height=100)

# Create a button for generating the SQL code
generate_button = st.button("Execute")

# Display a loading symbol while the SQL code is being generated
if generate_button:
    start_time = time.time()
    with st.spinner("Processing..."):
        sql_code, response = generate_sql_code(prompt)
        end_time = time.time()
        processing_time = end_time - start_time
        
        st.subheader("Generated SQL Code:")
        st.code(sql_code)
        
        st.subheader("Generated Response:")
        st.write(response)
        
        st.markdown(f"**Processing Time:** {processing_time:.2f} seconds")

        # Dynamic citation extraction and verification
        st.subheader("Source Information:")
        citations = re.findall(r'\[Source: [^\]]+\]', response)
        
        # Dynamically extract column information
        db_columns = extract_table_columns(sql_code)
        
        if db_columns:
            st.success(f"Columns extracted from the query: {', '.join(db_columns)}")
        else:
            st.error("No columns could be extracted from the SQL query. This might be due to an unexpected query structure.")
        
        if citations:
            unique_citations = list(set(citations))  # Remove duplicates
            st.write("Citations found in the response:")
            for cite in unique_citations:
                st.write(cite)
            
            # Verify citations
            cited_columns = [cite[8:-1].split('.')[-1] for cite in unique_citations]
            valid_citations = [col for col in cited_columns if col in db_columns or any(col in db_col for db_col in db_columns)]
            
            invalid_citations = set(cited_columns) - set(valid_citations)
        

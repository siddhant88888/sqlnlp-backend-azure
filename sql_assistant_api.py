from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_anthropic import ChatAnthropic
import boto3
import os
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for connection request body
class ConnectionRequest(BaseModel):
    db_uri: str
    llm_type: str
    api_key: str = None
    aws_access_key_id: str = None
    aws_secret_access_key: str = None

# Pydantic model for query request body
class QueryRequest(BaseModel):
    question: str
    db_uri: str
    llm_type: str
    api_key: str = None
    aws_access_key_id: str = None
    aws_secret_access_key: str = None

# Function to get database connection
def get_database_connection(db_uri):
    try:
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

# Function to get schema
def get_schema(db):
    return db.get_table_info()

# Function to clean SQL query
def clean(sql_query):
    return sql_query.replace('\n', ' ')

# Function to extract SQL query from LLM response
def extract_sql_query(response):
    # Look for SQL query between triple backticks
    match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If not found, look for the first SELECT statement
    match = re.search(r'\bSELECT\b.*', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    # If still not found, return the original response
    return response

@app.post("/connect")
async def connect(request: ConnectionRequest):
    logger.info(f"Received connection request for database URI: {request.db_uri}")
    try:
        db = get_database_connection(request.db_uri)
        schema = get_schema(db)
        logger.info("Successfully connected to the database and retrieved schema")
        return {"schema": schema}
    except Exception as e:
        logger.error(f"Failed to connect to the database: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Database connection failed: {str(e)}")

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        db = get_database_connection(request.db_uri)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {str(e)}")

    # Get the schema information
    schema = get_schema(db)

    # Set up LLM based on the request
    if request.llm_type == "OpenAI":
        if not request.api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required")
        os.environ["OPENAI_API_KEY"] = request.api_key
        llm = ChatOpenAI(temperature=0)
    elif request.llm_type == "AWS Bedrock":
        if not request.aws_access_key_id or not request.aws_secret_access_key:
            raise HTTPException(status_code=400, detail="AWS credentials are required")
        client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=request.aws_access_key_id,
            aws_secret_access_key=request.aws_secret_access_key,
            region_name="us-east-1"
        )
        llm = ChatBedrock(
            client=client,
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region="us-east-1"
        )
    elif request.llm_type == "Anthropic":
        if not request.api_key:
            raise HTTPException(status_code=400, detail="Anthropic API key is required")
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            api_key=request.api_key
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid LLM type")

    clean_sql = RunnableLambda(func=clean)

    template = """
    Based on the table schema below, write a SQL query or multiple SQL queries if necessary that would answer the user's question. 
    Make sure to handle complex questions that might require multiple queries or subqueries.
    If multiple queries are needed, separate them with semicolons.
    Provide ONLY the SQL query without any additional explanation.
    
    {schema}

    Question: {question} 
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)

    sql_chain = (
        RunnablePassthrough.assign(schema=lambda _: schema)
        | prompt 
        | llm.bind(stop="\nSQL Result:")
        | StrOutputParser()
        | clean_sql
    )

    # Generate the SQL query
    sql_query = sql_chain.invoke({"question": request.question})
    
    # Extract the actual SQL query
    extracted_sql_query = extract_sql_query(sql_query)

    template_response = """
    Based on the table schema, question, SQL query, and SQL response, write a detailed natural language response.
    Include relevant numbers, names, and any other specific information from the SQL response.
    If multiple queries were executed, summarize the results of all queries.

    {schema}

    Question: {question} 
    SQL query: {query} 
    SQL response: {response}

    Detailed Answer:
    """

    prompt_response = ChatPromptTemplate.from_template(template_response)

    def run_query(final_query): 
        queries = final_query.split(';')
        results = []
        for query in queries:
            if query.strip():
                results.append(db.run(query))
        return '\n'.join(results)

    full_chain = (
        RunnablePassthrough.assign(
            query=lambda _: extracted_sql_query,
            schema=lambda _: schema,
            response=lambda _: run_query(extracted_sql_query),
        )
        | prompt_response
        | llm
    )

    result = full_chain.invoke({"question": request.question})

    return {
        "sql_query": extracted_sql_query,
        "answer": result.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

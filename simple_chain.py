from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

api_key = os.getenv('OPENAI_KEY')

model = ChatOpenAI(model='gpt-4o-mini',
                   api_key=api_key)

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic} ',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'cricket'})

print(result)
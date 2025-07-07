from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('OPENAI_KEY')

model = ChatOpenAI(model='gpt-4o-mini',
                   api_key=api_key)

prompt1 = PromptTemplate(
    template='generate a detailed report on the {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='generate a five pointer summary from the following text\n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'unemployment in India'})

print(result)

chain.get_graph().print_ascii()
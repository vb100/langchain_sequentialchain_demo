import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain

api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(openai_api_key=api_key)

# ------------------ Chains ----------------------
template_1 = "Give a summary of this employee's performance review\n{review}"
prompt_1 = ChatPromptTemplate.from_template(template_1)
chain_1 = LLMChain(
    llm=llm,
    prompt=prompt_1,
    output_key='review_summary'
)

template_2 = "Identify key empoyee weaknesses in this review summary\n{review_summary}"
prompt_2 = ChatPromptTemplate.from_template(template_2)
chain_2 = LLMChain(
    llm=llm,
    prompt=prompt_2,
    output_key='weaknesses'
)

template_3 = "Create a personalized plan to help address and fix these weaknesses\n{weaknesses}"
prompt_3 = ChatPromptTemplate.from_template(template_3)
chain_3 = LLMChain(
    llm=llm,
    prompt=prompt_3,
    output_key='final_plan'
)

# -------- Build a Sequenctial Schain -----------
seq_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3],
    input_variables=['review'],
    output_variables=['review_summary', 'weaknesses', 'final_plan'],
    verbose=True
)

# ------------ Read Employee Review ---------------
with open('employee_review.txt') as f:
    employee_review = f.readlines()

# ---- Get the results (recommendation plan) ------
results = seq_chain(employee_review)

print(type(results))
print(results.keys)
print('--'*50)
print(results['final_plan'])


"""
Want to learn data science from industry experts?
CHECK IT: https://turingcollege.org/DataScienceGarage
"""

prompt_template = """
Use the following piece of information and answer the user's questions
If you don't know the answer, just say that you don't know the answer, don't try to make things up
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else
Helpful answer:
"""
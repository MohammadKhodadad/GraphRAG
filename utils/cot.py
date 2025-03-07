import os
import openai
from openai import OpenAI
# Once there was a boy, who saw his father smiling at him just before he passed away. He always sorrowed about that moment, until he saw his son standing by his deathbed.
# He just then realized, His father was also thinking about his own father. That was when he too smiled. 

class ChainOfThoughts:
    def __init__(self, api_key, retriever):
        """
        Initialize the OpenAIQA class with an API key.
        """
        self.client = OpenAI(api_key=api_key)
        self.retriever = retriever

    def query(self, query, documents=None):
        """
        Perform a RAG (Retrieval-Augmented Generation) query.

        Parameters:
        - query (str): The user's question.
        - documents (list of dict): A list of document dictionaries with content to augment the response.

        Returns:
        - str: The response from the OpenAI model.
        """
        if documents:
            context = "\n\n".join(documents)
            augmented_prompt = (
                f"Context:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "Please provide a relevant answer based on the above context. Do not use your internal knowledge. Answer just based on the context."
            )
        else:
            # If no documents are provided, use the query as is
            augmented_prompt = query

        # Send the augmented prompt to OpenAI's chat completion
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": augmented_prompt,
                    }
                ],
                model="o1",
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during query execution: {str(e)}"

    def summarize(self, query, pool):
        """
        Ask OpenAI what additional information is needed to answer the query.
        """
        pool_text = "\n\n".join(pool)
        # print("Extracted Pieces of Inforamtion:", pool_text)
        prompt = (
            f"You are given a set of information pieces and a query. The query is the question we want to answer and the pieces of information are all the information we have gathered.\n\n"
            f"Query: {query}\n\n"
            f"Pieces of Information:\n{pool_text}\n\n"
            f"Your job is to summarize all of the valuable information in the pieces of information we have that is related to the question. (remember some information might be related indirectly)"
            f"**Don't use your internal knowledge to answer a part of the question**: only look for what we need to know and what we have available in the paraphraphs."
            f"Return the summarized information as plain text."
        )
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o",
        )
        return response.choices[0].message.content.strip()
    def next_question(self, query, pool):
        """
        Ask OpenAI what additional information is needed to answer the query.
        """
        pool_text = "\n\n".join(pool)
        prompt = (
            f"You are given a set of information pieces and a query. The query is the question we want to answer and the pieces of information are all the information we have gathered.\n\n"
            f"Query: {query}\n\n"
            f"Pieces of Information:\n{pool_text}\n\n"
            f"Your job is to output the next best question based on the query and the available relevant information that we need to answer in order to answer the query or stop the process."
            f"**Don't use your internal knowledge to answer a part of the question**: only look for what we need to know and what we have available in the paraphraphs."
            f"**The generated question has to be complete by its own. Don't point to a part of the pieces of information or the query in the generated question. Simply include any information needed in the generated question itself."
            f"**You might find the next best question the query itself, when it is a small question itself.**"
            f"If the releveant paragraphs are enough to answer, return 'stop!' as plain text"
            f"Return only the question as plain text."
        )

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o",
        )
        return response.choices[0].message.content.strip()
    

    def chain_of_thoughts(self, query: str, max_iterations:int=5, top_k: int = 5, hybrid: bool=True, reranker:bool =True, reranker_top_k: int=3, summarize:bool=True):
        pool = []  # Start with an empty pool
        if max_iterations==0: # No Chain basically.
            retriever_result = self.retriever.similarity_search(query=query, top_k=top_k, hybrid=hybrid, reranker=reranker, reranker_top_k=reranker_top_k)
            top_paragraphs = retriever_result.get('reranker',
                                                retriever_result.get('documents',[[]])[0])
            for paragraph in top_paragraphs:
                if paragraph not in pool:
                    pool.append(paragraph)
            answer = self.query(query,pool)
            # print(f'Pool: \n{pool}')
            return {'answer':answer, 'retrived_information':pool}
        for _ in range(max_iterations):
            # Ask OpenAI what is needed
            next_question = self.next_question(query, pool)
            print(f"Next Question: { next_question}")
            if 'stop!' in next_question.lower():
                break
            
            retriever_result = self.retriever.similarity_search(query=next_question, top_k=top_k, hybrid=hybrid, reranker=reranker, reranker_top_k=reranker_top_k)
            top_paragraphs = retriever_result.get('reranker',
                                                retriever_result.get('documents',[[]])[0])
            # print('retriever results:',retriever_result)
            if not summarize:
                for paragraph in top_paragraphs:
                    if paragraph not in pool:
                        pool.append(paragraph)
            else:
                pool.append(self.summarize(next_question,top_paragraphs))
                print(f"Answer to the latest question: { pool[-1]}")
        answer = self.query(query,pool)
        # print(f'Pool: \n{pool}')
        return {'answer':answer, 'retrived_information':pool}
# Example usage
if __name__ == "__main__":
    import dotenv
    from retriever import Retriever
    dotenv.load_dotenv()
    ret=Retriever()
    ret.load_model()
    api_key = os.environ.get("OPENAI_API_KEY")  # Ensure this is set in your environment
    cot = ChainOfThoughts(api_key=api_key,retriever=ret)
    response = cot.chain_of_thoughts(query='What is the Haber process?')
    print("Response:", response)

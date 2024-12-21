import os
import openai
from openai import OpenAI
# Once there was a boy, who saw his father smiling at him just before he passed away. He always sorrowed about that moment, until he saw his son standing by his deathbed.
# He just then realized, His father was also thinking about his own father. That was when he too smiled. 

class OpenAIQA:
    def __init__(self, api_key):
        """
        Initialize the OpenAIQA class with an API key.
        """
        self.client = OpenAI(api_key=api_key)

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
            # Combine documents into a context string
            if 'reranker' in documents.keys():
                print("Reranker data used")
                context = "\n\n".join(documents.get("reranker", [['']])[0])
            else:
                print("documents data used")
                context = "\n\n".join(documents.get("documents", [['']])[0])
            augmented_prompt = (
                f"Context:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "Please provide a detailed and relevant answer based on the above context. Do not use your internal knowledge, just based on the documents."
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
                model="gpt-4o",
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during query execution: {str(e)}"


# Example usage
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")  # Ensure this is set in your environment
    openai_qa = OpenAIQA(api_key=api_key)

    query_text = "What are the key benefits of AI in healthcare?"
    documents ={"documents": ["AI can improve diagnostic accuracy by analyzing medical images.",
                               "AI-driven chatbots enhance patient engagement and support."]}

    response = openai_qa.query(query=query_text, documents=documents)
    print("Response:", response)

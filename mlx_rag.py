import argparse
from typing import Optional, Any
from mlx_lm import load, generate
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.llms.callbacks import llm_completion_callback 
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from pydantic import BaseModel

class OurLLM(CustomLLM, BaseModel):
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    def __init__(self, model_name: str, **data):
        super().__init__(**data)  # Initialize BaseModel part with data
        # Directly load the model and tokenizer
        self.model, self.tokenizer = load(model_name)
    context_window: int = 2096
    max_tokens : int = 100
    model_name: str = "custom"
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window = self.context_window,
            model_name=self.model_name,
            max_tokens=self.max_tokens
        )

    def process_generated_text(self, text: str) -> str:
        token_pos = text.find("\n\n")
        if token_pos != -1:
            # Truncate text at the first occurrence of two new lines
            return text[:token_pos]
        return text

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
       # Remove 'formatted' argument if present
        kwargs.pop('formatted', None)
    
        generated_text = generate(self.model, self.tokenizer, prompt=prompt, verbose=False, **kwargs)
        processed_text = self.process_generated_text(generated_text)
        return CompletionResponse(text=processed_text)


    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        generated_text = generate(self.model, self.tokenizer, prompt=prompt, verbose=False, **kwargs)
        processed_text = self.process_generated_text(generated_text)
        for char in processed_text:  
            yield CompletionResponse(text=char, delta=char)


def main():
    parser = argparse.ArgumentParser(description="Query a document collection with an LLM.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to use for the LLM.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing the documents to index.")
    parser.add_argument("--embed_model", type=str, required=True, help="Embed model to use for vectorizing documents.")
    parser.add_argument("--query", type=str, required=True, help="Query to perform on the document collection.")

    args = parser.parse_args()

    # Convert input from the user into strings
    model_name = str(args.model_name)
    directory = str(args.directory)
    embed_model = str(args.embed_model)
    query_str = str(args.query)

    # Setup the LLM and embed model
    Settings.llm = OurLLM(model_name=model_name)
    Settings.embed_model = embed_model

    # Load documents and create index
    documents = SimpleDirectoryReader(directory).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    # Perform query and print response
    query_engine = index.as_query_engine()
    response = query_engine.query(query_str)
    print(response)


if __name__ == "__main__":
    main()

#Usage: python mlx_rag.py --model_name "mlx-community/Mistral-7B-v0.1-hf-4bit-mlx" --directory "data" --embed_model "local:BAAI/bge-base-en-v1.5" --query "Complete the sentence: In all criminal prosecutions, the accused shall enjoy"

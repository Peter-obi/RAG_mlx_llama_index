# RAG_llms_mlx_llama_index
Run Retrival Augmented Generation using local mlx models and llama index
#### Setting Up the Environment

```
conda create -n rag_mlx python=3.9
conda activate rag_mlx
pip install -r requirements.txt
```
You can use the notebook and switch out the models there for your choice. Documents to be 'raged' on go into the data folder. 

#### Run in terminal
```
python mlx_rag.py --model_name "mlx-community/Mistral-7B-v0.1-hf-4bit-mlx" --directory "data" --embed_model "local:BAAI/bge-base-en-v1.5" --query "Complete the sentence: In all criminal prosecutions, the accused shall enjoy"
```
Once you get this running, you can always build more complex RAG pipelines and set up agents!

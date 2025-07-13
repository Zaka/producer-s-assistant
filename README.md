# Producer's Assistant

An AI-powered assistant that answers music production questions using LangChain and Elasticsearch.

## Features
- Natural language Q&A
- Real-time search through production knowledge
- Coming soon: real-world scraping and dataset

## Stack
- LangChain
- Elasticsearch
- OpenAI Embeddings

## How to run it

Install all the requirements

  pip install -r requirements.txt

If first time deploying elasticsearch docker:

docker run -d --name elastic \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.2

If the container already deployed:

docker container start /elastic

If first time populating elasticsearch db or after a dataset update:

cd app/
python ingest_data.py

Finally, to run the assistant:

cd app/
python qa_pipeline.py

To launch the Streamlit web interface instead:

cd app/
streamlit run streamlit_app.py

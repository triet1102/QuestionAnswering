# Install python requirements
```bash
pip install -r requirements/res.txt --no-deps
```

# Install and launch ElasticSearch locally (for macOS)
Source from [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html). Install elasticsearch 7 to be compatible with haystack.

```bash
docker network create elastic
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.11.3
docker run --name es01 --net elastic -p 9200:9200 -it -m 1GB docker.elastic.co/elasticsearch/elasticsearch:8.11.3

# Then copy the generated password and export
export ELASTIC_PASSWORD="your_password"

# Copy the http_ca.crt SSL certificate from the container to your local machine
docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .

# Make a REST API call to Elasticsearch to ensure the Elasticsearch container is running
curl --cacert http_ca.crt -u elastic:$ELASTIC_PASSWORD https://localhost:9200
```

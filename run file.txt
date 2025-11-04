
az login
az acr login --name projectaipictcontainerregistry

docker build . -t projectaipictcontainerregistry.azurecr.io/projectai-pict/api:v4

docker push projectaipictcontainerregistry.azurecr.io/projectai-pict/api:v4
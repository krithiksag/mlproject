## End to End Machine Learning Project with aws deployment
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

Configure EC2 as self-hosted runner:
Setup github secrets:
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = eu-north-1

AWS_ECR_LOGIN_URI = 676206924952.dkr.ecr.eu-north-1.amazonaws.com

ECR_REPOSITORY_NAME = studentperformance

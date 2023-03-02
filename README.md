# News Recommender

Project created with MLOps-Template cookiecutter. For more info: https://mlopsstudygroup.github.io/mlops-guide/
Create personalized UX with news recommendations

This repo will hold the backend for the news recommendation.

Initially the algorithm will be perfected with the well known movie lens data. We will use collaborative filtering to produce predicted ratings and make recommendation ranked by their predicted ratings for the users in the dataset.

Later this process will be repeated in the actual news recommendation but with hybrid approach as collaborative filtering suffers from cold start. Also, we have some idea of preference of user collected as their interests with news tags.

Wish us luck!!!!!!!


## 📋 Requirements

* DVC
* Python3 and pip

## 🏃🏻 Running Project


#### MacOS and Linux
Setup your credentials on ```~/.aws/credentials``` and ```~/.aws/config```. DVC works perfectly with IBM Obejct Storage, although it uses S3 protocol, you can also see this in other portions of the repository.


~/.aws/credentials

```credentials
[default]
aws_access_key_id = {Key ID}
aws_secret_access_key = {Access Key}
```


### ✅ Pre-commit Testings

In order to activate pre-commit testing you need ```pre-commit```

Installing pre-commit with pip
```
pip install pre-commit
```

Installing pre-commit on your local repository. Keep in mind this creates a Github Hook.
```
pre-commit install
```

Now everytime you make a commit, it will run some tests defined on ```.pre-commit-config.yaml``` before allowing your commit.

**Example**
```
$ git commit -m "Example commit"

black....................................................................Passed
pytest-check.............................................................Passed
```


### ⚗️ Using DVC

Download data from the DVC repository(analog to ```git pull```)
```
dvc pull
```

Reproduces the pipeline using DVC
```
dvc repro
```

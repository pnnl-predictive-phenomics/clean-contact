# clean-contact-website

Create a new conda environment, install required packages

```bash
conda create -n clean-contact-website python=3.11 -y
conda activate clean-contact-website
python -m pip install Django djangorestframework markdown django-filter
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers
python -m pip install scipy scikit-learn pandas biotite
```

Follow tutorial from [this link]([https://docs.djangoproject.com/en/5.1/howto/deployment/](https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/modwsgi/) to deploy website using Apache and mod_wsgi, do the [checklist](https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/), change secret keys in Django setting, and set up `SSL`. 

***app keys and email app passwords have been redacted***
***large files (emb_train.pt and train.csv) under clean-contact-website-backend/server/apps/cleancontact have been replaced with empty files since my git lfs quota is used up. to get the two files, provide me your email so that I can share them with you***

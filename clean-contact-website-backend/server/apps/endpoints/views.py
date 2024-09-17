import os

from django.shortcuts import render
from django.db import transaction
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.conf import settings
from django.core.mail import send_mail
from rest_framework.exceptions import APIException

# Create your views here.
from rest_framework import viewsets
from rest_framework import mixins

from apps.endpoints.forms import RunMLForm
from apps.cleancontact.main import CLEANContact

from apps.endpoints.models import Endpoint
from apps.endpoints.serializers import EndpointSerializer

from apps.endpoints.models import MLAlgorithm
from apps.endpoints.serializers import MLAlgorithmSerializer

from apps.endpoints.models import MLAlgorithmStatus
from apps.endpoints.serializers import MLAlgorithmStatusSerializer

from apps.endpoints.models import MLRequest
from apps.endpoints.serializers import MLRequestSerializer

import smtplib
from email.mime.text import MIMEText

import warnings

class EndpointViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()


class MLAlgorithmViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = MLAlgorithmSerializer
    queryset = MLAlgorithm.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm = instance.parent_mlalgorithm,
                                                        created_at__lt=instance.created_at,
                                                        active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
    MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])

class MLAlgorithmStatusViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.CreateModelMixin
):
    serializer_class = MLAlgorithmStatusSerializer
    queryset = MLAlgorithmStatus.objects.all()
    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                # set active=False for other statuses
                deactivate_other_statuses(instance)



        except Exception as e:
            raise APIException(str(e))

class MLRequestViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.UpdateModelMixin
):
    serializer_class = MLRequestSerializer
    queryset = MLRequest.objects.all()

def index(request):
    if request.method == 'POST':
        warnings.warn('request post')
        form = RunMLForm(request.POST)
        if form.is_valid():
            warnings.warn('form is valid')
            uid = form.cleaned_data['uid']
            pdbsource = form.cleaned_data['pdb']
            user = form.cleaned_data['email'].replace('@', '.')

            cc = CLEANContact()

            if not pdbsource == '':
                if not os.path.exists('/var/www/.cache/useruploads'):
                    os.makedirs('/var/www/.cache/useruploads')

                with open(f'/var/www/.cache/useruploads/{user}_{uid}.pdb', 'w') as f:
                    f.write(form.cleaned_data['pdb'])

                results = cc.run(
                    uniprot_id=uid,
                    pdb_file=f'/var/www/.cache/useruploads/{user}_{uid}.pdb'
                )

            else:
                results = cc.run(
                    uniprot_id=uid,
                )

            print('results', results)
            if results['status'] == 'success':
                pval = results['pval_results'][0][1].split('/')[0]
                maxsep = results['maxsep_results'][0][1].split('/')[0]
                seq = results['seq']
                subject = f'Do Not Reply: CLEAN-Contact Results - {uid}'
                message = f'Here are the CLEAN-Contact prediction results for your submission:\n\n\
                    UniProt ID: {uid} \n\
                    Sequence: {seq} \n\
                    P-value selection: {pval} \n\
                    Max-separation selection: {maxsep} \n\n\
                    Do not reply to this email as the mailbox is not monitored. \n\
                    For result or usage issues, please contact the first or corresponding authors of the paper.'
            else:
                subject = f'Do Not Reply: CLEAN-Contact Results - {uid}'
                message = f'Unfortunately, we cannot obtain a result for your input. \n\
This may be caused by non amino acid characters in the input sequence, wrong UniProt ID, or no available AlphaFold 2 predicted structure for the given protein. \n\
Please forward this email to the first author without any truncation for troubleshooting purpose, we will try to get back to you as soon as we can. \n\n\
UniProt ID: {uid}\n\
Do not reply to this email as the mailbox is not monitored. \n\
For result or usage issues, please contact the first or corresponding authors of the paper.'
            sender = 'cleancontactresults@gmail.com'
            recipient = form.cleaned_data['email']
            password = 'EMAIL_APP_PASSWORD_REDACTED'
            
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = sender
            msg['To'] = recipient
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
                smtp_server.login(sender, password)
                smtp_server.sendmail(sender, recipient, msg.as_string())
                print('Email sent')
            # send_mail(subject, message, settings.EMAIL_HOST_USER, [form.cleaned_data['email']])
        else:
            warnings.warn('form is invalid')

    else:
        warnings.warn('other request')
        form = RunMLForm()
    # template = loader.get_template('index.html')

    return render(request, "index.html")


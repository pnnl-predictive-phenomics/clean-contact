"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

import inspect
from apps.cleancontact.registry import MLRegistry
from apps.cleancontact.main import CLEANContact

try:
    registry = MLRegistry() # create ML registry
    cc = CLEANContact()
    # add to ML registry
    registry.add_algorithm(endpoint_name="clean_contact",
                            algorithm_object=cc,
                            algorithm_name="clean contact",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Yang, Yuxin",
                            algorithm_description="CLEAN-Contact for EC number prediction",
                            algorithm_code=inspect.getsource(CLEANContact))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))

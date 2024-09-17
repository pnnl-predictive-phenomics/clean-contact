from django.test import TestCase
from apps.cleancontact.registry import MLRegistry
from apps.cleancontact.main import CLEANContact
import inspect

class CleanContactTests(TestCase):
    def test_clean_contact(self):
        clean_contact = CLEANContact()
        
        input_data = {
            "uniprot_id": "P70994",
            "pdb_file": None,
        }

        response = clean_contact.run(**input_data)

        print('response', response)
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["message"], "Predictions generated successfully")
        self.assertTrue("pval_results" in response)
        self.assertTrue("pval_confidence_results" in response)
        self.assertTrue("maxsep_results" in response)
        self.assertTrue("maxsep_confidence_results" in response)

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "clean_contact"
        algorithm_object = CLEANContact()
        algorithm_name = "clean contact"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Yuxin Yang"
        algorithm_description = "CLEAN-Contact for EC number prediction"
        algorithm_code = inspect.getsource(CLEANContact)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)

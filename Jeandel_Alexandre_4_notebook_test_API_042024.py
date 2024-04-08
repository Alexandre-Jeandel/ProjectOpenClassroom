# Import packages
import unittest
import requests
import json

# Global variables definition
api_url = "http://localhost:5000/"
n_clients = 1000

# API tests
class TestDefaulPredictAPI(unittest.TestCase):

    # Test # 1
    def test_welcome(self):

        response = requests.get(api_url)

        # Test # 1.1: status code = 200
        self.assertEqual(response.status_code, 200)

        # Test # 1.2: The response gives the right message
        real_res = json.loads(response.content)
        expected_res = "Hello world! Welcome to the Default Predictor API!"
        self.assertEqual(real_res, expected_res)
    
    # Test # 2
    def test_load_client_list(self):
        
        response = requests.get(api_url + '/client_list')

        # Test # 2.1: status code = 200
        self.assertEqual(response.status_code, 200)

        # Test # 2.2: the output is a list
        id_list = response.json()
        self.assertIsInstance(id_list, list)

        # Test # 2.3: the output list has the right lenght
        len_id_list = len(id_list)
        self.assertEqual(len_id_list, n_clients)

        # Test # 2.4: the first id in the list is the right one
        first_id = 100001
        self.assertEqual(id_list[0], first_id)
    
    # Test # 3
    def test_load_client(self):
        
        client_id = 100028
        response = requests.get(api_url + f'/client?id={client_id}')
        
        # Test # 3.1: status code = 200
        self.assertEqual(response.status_code, 200)

        # Test # 3.2: the client data are correct
        client = response.json()
        self.assertEqual(client['AMT_ANNUITY'], 14220.0)
        self.assertEqual(client['AMT_CREDIT'], 1575000.0)
        self.assertEqual(client['AMT_INCOME_TOTAL'], 315000.0)
        self.assertEqual(client['DAYS_BIRTH'], 13976.0)

    # Test # 4
    def test_load_data(self):
        
        col = 'DAYS_BIRTH'
        response = requests.get(api_url + f'/data?col={col}')
        
        # Test # 4.1: status code = 200
        self.assertEqual(response.status_code, 200)

        # Test # 4.2: the output is a list
        age_list = response.json()
        self.assertIsInstance(age_list, list)

        # Test # 4.3: the output list has the right lenght
        len_age_list = len(age_list)
        self.assertEqual(len_age_list, n_clients)

        # Test # 4.4: the first id in the list is the right one
        first_age = 19241
        self.assertEqual(age_list[0], first_age)

    # Test # 5
    def test_predict_defaul(self):

        client_id = 100005
        response = requests.get(api_url + f'/predict_default?id_client={client_id}')

        # Test # 5.1: status code = 200
        # self.assertEqual(response.status_code, 200)

        # Test # 5.2: the ptrediction is correct (Once the model re-compiled, test values have to be updated)
        prediction = response.json()
        self.assertEqual(prediction['proba_0'], 0.21369841008166157)
        self.assertEqual(prediction['proba_1'], 0.7863015899183384)

if __name__ == '__main__':
    unittest.main()
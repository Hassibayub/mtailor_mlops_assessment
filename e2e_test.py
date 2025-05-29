import requests
import os

def test_predict_endpoint():
    # API endpoint
    url = "https://api.cortex.cerebrium.ai/v4/p-9d0ef0c4/app/predict"
    
    # Headers for authentication
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTlkMGVmMGM0IiwiaWF0IjoxNzQ4NTIxMzg3LCJleHAiOjIwNjQwOTczODd9.NNvAY9lY_HLcVX3p2OBYVl1ZlPIobH0dpY7Zo5-oia_Tf4a9D7l5Pn0dXfD1YwYvHpJSxYDKHaYb1Rm_MPJ7d7NYbvyz3hUCWwtNb3B4R0VAEjHeUotVLRl6sQPnE0jNkiLz5M69h4nAeewI5LcYasl2V78HCNJIY3Ppf9qz7O1--dRZLTWvhaN6axD7c-O6sGIDyRwEwiEl5Y1WmlRla11DPadwi42c3D1yEqXB1GSKwpTizPMAsUi1MgnKDZi-51wD2ZxsAcQ77ZnPQtJJMYQ7JtdfafiuwTu9nlTfVxVINrgOPoHWUloO8xCksyy_F0oRJEJfmnp_gCw3qGAzvQ'
    }
    
    # Path to test image file
    test_image_path = "n01667114_mud_turtle.JPEG"  # Make sure this image exists in your directory
    
    # Prepare the file for upload
    with open(test_image_path, 'rb') as image_file:
        files = {
            'file': ('test_image.jpg', image_file, 'image/jpeg')
        }
        
        # Make POST request
        response = requests.post(url, headers=headers, files=files)
        
        # Print response
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Basic assertions
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert "class_id" in response.json()["prediction"]
        assert "confidence" in response.json()["prediction"]
        assert "inference_time" in response.json()

if __name__ == "__main__":
    test_predict_endpoint()
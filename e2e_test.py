import requests
import os
import sys

# Base URL for all API endpoints
BASE_URL = "https://api.cortex.cerebrium.ai/v4/p-9d0ef0c4/app"

# Headers for authentication
HEADERS = {
    'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTlkMGVmMGM0IiwiaWF0IjoxNzQ4NTIxMzg3LCJleHAiOjIwNjQwOTczODd9.NNvAY9lY_HLcVX3p2OBYVl1ZlPIobH0dpY7Zo5-oia_Tf4a9D7l5Pn0dXfD1YwYvHpJSxYDKHaYb1Rm_MPJ7d7NYbvyz3hUCWwtNb3B4R0VAEjHeUotVLRl6sQPnE0jNkiLz5M69h4nAeewI5LcYasl2V78HCNJIY3Ppf9qz7O1--dRZLTWvhaN6axD7c-O6sGIDyRwEwiEl5Y1WmlRla11DPadwi42c3D1yEqXB1GSKwpTizPMAsUi1MgnKDZi-51wD2ZxsAcQ77ZnPQtJJMYQ7JtdfafiuwTu9nlTfVxVINrgOPoHWUloO8xCksyy_F0oRJEJfmnp_gCw3qGAzvQ'
}

def test_root_endpoint():
    """Test the root endpoint to check if API is running"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/", headers=HEADERS)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Assertions
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["message"] == "MTailor Model API is running"
        print("✅ Root endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Root endpoint test failed: {str(e)}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", headers=HEADERS)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text.lower()}")
        
        res = response.text.lower()
        print("response:", res)
        
        # Assertions
        assert response.status_code == 200
        assert res == "ok"
        
        
        print("✅ Health endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Health endpoint test failed: {str(e)}")
        return False

def test_ready_endpoint():
    """Test the ready endpoint"""
    print("\nTesting ready endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/ready", headers=HEADERS)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Assertions
        print(response.text)
        
        assert response.status_code == 200
        assert response.text == "OK"
        print("✅ Ready endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Ready endpoint test failed: {str(e)}")
        return False

def test_predict_endpoint():
    """Test the predict endpoint with an image"""
    print("\nTesting predict endpoint...")
    
    # Path to test image file
    test_image_path = "n01667114_mud_turtle.JPEG"  # Make sure this image exists in your directory
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    try:
        # Prepare the file for upload
        with open(test_image_path, 'rb') as image_file:
            files = {
                'file': ('test_image.jpg', image_file, 'image/jpeg')
            }
            
            # Make POST request
            response = requests.post(f"{BASE_URL}/predict", headers=HEADERS, files=files)
            
            # Print response
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            
            # Basic assertions
            assert response.status_code == 200
            assert "prediction" in response.json()
            assert "class_id" in response.json()["prediction"]
            assert "confidence" in response.json()["prediction"]
            assert "inference_time" in response.json()
            assert "status" in response.json() and response.json()["status"] == "success"
            
            print("✅ Predict endpoint test passed")
            return True
    except Exception as e:
        print(f"❌ Predict endpoint test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all endpoint tests and report results"""
    print("Starting E2E API tests...")
    
    tests = [
        # ("Root Endpoint", test_root_endpoint),
        ("Health Endpoint", test_health_endpoint),
        ("Ready Endpoint", test_ready_endpoint),
        # ("Predict Endpoint", test_predict_endpoint)
    ]
    
    results = []
    for name, test_func in tests:
        results.append((name, test_func()))
    
    # Print summary
    print("\n=== TEST RESULTS SUMMARY ===")
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed! 🎉")
        return 0
    else:
        print("\nSome tests failed. 😢")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
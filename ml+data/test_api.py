import requests
import json

def test_api():
    """Test the Flask API to make sure it works"""
    
    base_url = "http://localhost:5001"
    
    print("Testing SafeInbox Email Phishing Detection API")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test 2: Test with a normal email
    print("\n2. Testing legitimate email...")
    normal_email = {
        "subject": "Meeting reminder for tomorrow",
        "body": "Duckyluck Casino Hello emailagamjotsingh,(1) FINAL MESSAGE: Best Mobile Casino: 80 Free Spins!, You could win the Millionaire's Life!. Duckyluck Casino Confirm Your Info... CHECK ID 8273897060 DATE Mon,28 Jul-2025 PAYOUT $5000.00 Confirm Here Your account information: Name: emailagamjotsingh Email: emailagamjotsingh@gmail.com Payout Verification : $5000.00 PAYOUT Coupon Code : CASINO400 To be removed from our list simply click here or write to us at : 8754 Windfall,,Drive Pawtucket,RI,02860",
        "from": "manager@company.com"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=normal_email)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Is Phishing: {result.get('isPhishing')}")
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
    except Exception as e:
        print(f"Normal email test failed: {e}")
    
    # Test 3: Test with a suspicious email
    print("\n3. Testing suspicious email...")
    suspicious_email = {
        "subject": "URGENT: Your account will be suspended!! Act now!",
        "body": "Congratulations! You have won $1,000,000! Click here immediately to claim your prize before it expires in 24 hours. Update your payment information now or lose this opportunity forever! Verify your account details at http://bit.ly/fake-bank urgent action required!",
        "from": "noreply@suspicious-bank.com"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=suspicious_email)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Is Phishing: {result.get('isPhishing')}")
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
    except Exception as e:
        print(f"Suspicious email test failed: {e}")
    
    # Test 4: Test Chrome extension format
    print("\n4. Testing Chrome extension format...")
    chrome_test = {
        "subject": "Free Bitcoin! Click now!",
        "body": "You have received free bitcoin! Click this link now to claim your cryptocurrency. Limited time offer expires today!",
        "from": "bitcoin@scammer.com"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=chrome_test)
        print(f"Status: {response.status_code}")
        result = response.json()
        print("Chrome extension expects:")
        print(f"  - isPhishing: {result.get('isPhishing')} (True/False)")
        print(f"  - prediction: {result.get('prediction')} (0 or 1)")
        print(f"  - timestamp: {result.get('timestamp')}")
    except Exception as e:
        print(f"Chrome extension format test failed: {e}")

# Run the tests
if __name__ == "__main__":
    test_api() 
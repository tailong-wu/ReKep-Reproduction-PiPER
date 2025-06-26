import requests
import json
import time

# Load the action sequence from the JSON file
with open('outputs/action.json', 'r') as f:
    action_data = json.load(f)
ee_action_seq = action_data['ee_action_seq']

# API endpoint URL
url = "http://localhost:8000/solve-ik/"

# Prepare the request data
request_data = {"ee_action_seq": ee_action_seq}

# --- Start Timer ---
start_time = time.time()

# Send the POST request
response = requests.post(url, json=request_data)

# --- End Timer ---
end_time = time.time()

# Check the response and print the result
if response.status_code == 200:
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Request completed in {elapsed_time:.4f} seconds.")

    result = response.json()
    print("\nSuccessfully received joint angles:")
    # To make it more readable, let's pretty-print the JSON
    print(json.dumps(result, indent=4))

    # You can also save it to a file if needed
    with open('joint_angles_from_api.json', 'w') as f:
        json.dump(result, f, indent=4)
    print("\nSaved results to joint_angles_from_api.json")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
import cv2 
import numpy as np
from flask import Flask, request, render_template, jsonify
import base64

app = Flask(__name__)

def check_currency(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check for the presence of the number 2000
    text = cv2.imread('2000_text.png', 0)
    result = cv2.matchTemplate(gray, text, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    # Check for the presence of the Gandhi portrait
    gandhi = cv2.imread('gandhi_portrait.png', 0)
    result = cv2.matchTemplate(gray, gandhi, cv2.TM_CCOEFF_NORMED)
    _, max_val2, _, _ = cv2.minMaxLoc(result)
    
    # Check for the correct color (predominantly purple for ₹2000 note)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    purple_lower = np.array([130, 50, 50])
    purple_upper = np.array([170, 255, 255])
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
    purple_ratio = np.sum(purple_mask) / (image.shape[0] * image.shape[1])
    
    # Make a decision based on the checks
    if max_val > 0.8 and max_val2 > 0.8 and purple_ratio > 0.3:
        return True
    else:
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the image from the POST request
        file = request.files['image']
        # Read the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Check the currency
        result = check_currency(image)
        
        # Encode the image to base64 for displaying in HTML
        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'result': 'Likely genuine' if result else 'Possibly fake',
            'image': img_str
        })
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

print("Flask app is ready to run. Make sure to create the necessary template and reference images.")
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from functools import wraps

app = Flask(__name__)
# Secret key for session
app.secret_key = os.urandom(24)

# Sample user credentials - in a real app, use a database with hashed passwords
users = {
    "admin": "admin123",
    "support": "support123"
}

# Load the dataset
df = pd.read_excel("Online Retail.xlsx")

# Drop rows with missing CustomerID, Description, or InvoiceDate
df.dropna(subset=['CustomerID', 'Description', 'InvoiceDate'], inplace=True)

# Convert CustomerID to integer and InvoiceDate to datetime
df['CustomerID'] = df['CustomerID'].astype(int)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Encode product descriptions
encoder = LabelEncoder()
df['DescriptionEncoded'] = encoder.fit_transform(df['Description'])

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# Function to get sorted customer products
def get_customer_products(customer_id, sort_by="date", order="asc"):
    customer_data = df[df['CustomerID'] == customer_id][['Description', 'InvoiceDate']].drop_duplicates()

    if customer_data.empty:
        return None

    if sort_by == "name":
        customer_data = customer_data.sort_values(by="Description", ascending=(order == "asc"))
    else:
        customer_data = customer_data.sort_values(by="InvoiceDate", ascending=(order == "asc"))

    return customer_data.to_dict(orient='records')

# Function to return support options
def get_support_options(issue):
    options = {
        "DAMAGED ITEM": ["Request Refund", "Replace Item", "Contact Support"],
        "MISSING PARTS": ["Send Replacement", "Report Issue"],
        "WRONG ITEM SENT": ["Return Item", "Request Correct Product"],
        "GENERAL QUERY": ["Track Order", "Talk to a Representative"]
    }
    return options.get(issue, ["Talk to a Representative"])

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_page'))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'GET':
        return render_template('login.html')
    
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if username in users and users[username] == password:
        session['username'] = username
        return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "Invalid username or password"})

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login_page'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html', username=session.get('username'))

@app.route('/get_products', methods=['POST'])
@login_required
def get_products():
    data = request.json
    customer_id = int(data['customer_id'])
    sort_by = data['sort_by']
    order = data['order']

    products = get_customer_products(customer_id, sort_by, order)
    
    if products is None:
        return jsonify({"error": "No records found"})

    return jsonify(products)

@app.route('/get_support_options', methods=['POST'])
@login_required
def support_options():
    data = request.json
    issue = data['issue']
    options = get_support_options(issue)
    return jsonify(options)

if __name__ == '__main__':
    app.run(debug=True)
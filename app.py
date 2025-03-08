from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from mlxtend.frequent_patterns import apriori, association_rules
import calendar
import numpy as np

app = Flask(__name__)

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

# Create month and year columns for time-based analysis
df['Month'] = df['InvoiceDate'].dt.month
df['Year'] = df['InvoiceDate'].dt.year

# Function to get customers with negative quantities (returns)
def get_customers_with_returns(min_returns=5):
    # Group by customer and count negative quantity orders
    customer_returns = df[df['Quantity'] < 0].groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',  # Count unique invoices with returns
        'Quantity': lambda x: abs(x.sum()),  # Total items returned
        'InvoiceDate': lambda x: x.max()  # Most recent return
    }).reset_index()
    
    # Rename columns for clarity
    customer_returns.columns = ['CustomerID', 'ReturnInvoices', 'ItemsReturned', 'LastReturnDate']
    
    # Filter customers with minimum returns
    flagged_customers = customer_returns[customer_returns['ReturnInvoices'] >= min_returns]
    
    # Sort by number of returns (descending)
    return flagged_customers.sort_values('ReturnInvoices', ascending=False).to_dict(orient='records')

# Function to get top-selling products by month and year
def get_top_products_by_month(year, month, top_n=10):
    # Filter by year and month
    monthly_data = df[(df['Year'] == year) & (df['Month'] == month) & (df['Quantity'] > 0)]
    
    if monthly_data.empty:
        return []
    
    # Group by product and sum quantities
    product_sales = monthly_data.groupby('Description').agg({
        'Quantity': 'sum',
        'StockCode': 'first'  # Keep the stock code for reference
    }).reset_index()
    
    # Sort by quantity sold and get top N
    top_products = product_sales.sort_values('Quantity', ascending=False).head(top_n)
    
    # Return as list of dictionaries
    return top_products.to_dict(orient='records')

# Function to generate product combinations for recommendations
def get_product_recommendations(min_support=0.01, min_confidence=0.3):
    # Create a basket matrix (one-hot encoded)
    basket = df[df['Quantity'] > 0].groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0)
    basket = basket.set_index('InvoiceNo')
    
    # Convert quantities to binary (1 if product was purchased, 0 otherwise)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
    # No rules can be generated if frequent_itemsets is empty
    if frequent_itemsets.empty:
        return []
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Sort by lift (measure of association strength)
    rules = rules.sort_values('lift', ascending=False)
    
    # Format results
    recommendations = []
    for _, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        recommendations.append({
            'if_purchased': antecedents,
            'recommend': consequents,
            'confidence': round(row['confidence'] * 100, 2),
            'lift': round(row['lift'], 2)
        })
    
    return recommendations[:20]  # Return top 20 recommendations

# Original function to get customer products
def get_customer_products(customer_id, sort_by="date", order="asc"):
    customer_data = df[df['CustomerID'] == customer_id][['Description', 'InvoiceDate']].drop_duplicates()

    if customer_data.empty:
        return None

    if sort_by == "name":
        customer_data = customer_data.sort_values(by="Description", ascending=(order == "asc"))
    else:
        customer_data = customer_data.sort_values(by="InvoiceDate", ascending=(order == "asc"))

    return customer_data.to_dict(orient='records')

# Original function to return support options
def get_support_options(issue):
    options = {
        "DAMAGED ITEM": ["Request Refund", "Replace Item", "Contact Support"],
        "MISSING PARTS": ["Send Replacement", "Report Issue"],
        "WRONG ITEM SENT": ["Return Item", "Request Correct Product"],
        "GENERAL QUERY": ["Track Order", "Talk to a Representative"]
    }
    return options.get(issue, ["Talk to a Representative"])

# Get available months and years in the dataset
def get_available_time_periods():
    time_periods = df[['Year', 'Month']].drop_duplicates().sort_values(['Year', 'Month'])
    result = []
    for _, row in time_periods.iterrows():
        result.append({
            'year': int(row['Year']),
            'month': int(row['Month']),
            'month_name': calendar.month_name[int(row['Month'])]
        })
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_flagged_customers', methods=['POST'])
def flagged_customers():
    data = request.json
    min_returns = int(data.get('min_returns', 5))
    customers = get_customers_with_returns(min_returns)
    return jsonify(customers)

@app.route('/get_top_products', methods=['POST'])
def top_products():
    data = request.json
    year = int(data.get('year'))
    month = int(data.get('month'))
    top_n = int(data.get('top_n', 10))
    products = get_top_products_by_month(year, month, top_n)
    return jsonify(products)

@app.route('/get_product_recommendations', methods=['POST'])
def product_recommendations():
    data = request.json
    min_support = float(data.get('min_support', 0.01))
    min_confidence = float(data.get('min_confidence', 0.3))
    recommendations = get_product_recommendations(min_support, min_confidence)
    return jsonify(recommendations)

@app.route('/get_time_periods', methods=['GET'])
def time_periods():
    periods = get_available_time_periods()
    return jsonify(periods)

# Keep original routes
@app.route('/get_products', methods=['POST'])
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
def support_options():
    data = request.json
    issue = data['issue']
    options = get_support_options(issue)
    return jsonify(options)

if __name__ == '__main__':
    app.run(debug=True)
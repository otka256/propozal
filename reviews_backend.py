#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Files to store reviews
CSV_FILE = 'customer_reviews.csv'
TXT_FILE = 'customer_reviews.txt'

@app.route('/api/reviews', methods=['POST'])
def add_review():
    """Add a new review and save to CSV file"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'rating', 'review']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'فیلد {field} الزامی است'}), 400
        
        # Create review data
        review_data = {
            'تاریخ': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'نام': data['name'].strip(),
            'شرکت': data.get('company', 'کارفرما').strip(),
            'امتیاز': int(data['rating']),
            'نظر': data['review'].strip()
        }
        
        # Save to CSV file
        csv_file = 'customer_reviews.csv'
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', encoding='utf-8', newline='') as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=['تاریخ', 'نام', 'شرکت', 'امتیاز', 'نظر'])
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write review data
            writer.writerow(review_data)
        
        # Also save to text file for easy reading
        txt_file = 'customer_reviews.txt'
        with open(txt_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"تاریخ: {review_data['تاریخ']}\n")
            f.write(f"نام: {review_data['نام']}\n")
            f.write(f"شرکت: {review_data['شرکت']}\n")
            f.write(f"امتیاز: {'⭐' * review_data['امتیاز']} ({review_data['امتیاز']}/5)\n")
            f.write(f"نظر: {review_data['نظر']}\n")
            f.write(f"{'='*60}\n")
        
        logger.info(f"New review saved: {review_data['نام']} - {review_data['امتیاز']} stars")
        
        return jsonify({
            'success': True,
            'message': 'نظر شما با موفقیت ثبت شد'
        })
            
    except Exception as e:
        logger.error(f"Error adding review: {e}")
        return jsonify({'error': 'خطا در پردازش درخواست'}), 500

@app.route('/api/reviews/stats', methods=['GET'])
def get_review_stats():
    """Get simple review statistics"""
    try:
        csv_file = 'customer_reviews.csv'
        
        if not os.path.exists(csv_file):
            return jsonify({
                'success': True,
                'total_reviews': 0,
                'average_rating': 0
            })
        
        import csv
        total_reviews = 0
        total_rating = 0
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_reviews += 1
                total_rating += int(row['امتیاز'])
        
        average_rating = round(total_rating / total_reviews, 1) if total_reviews > 0 else 0
        
        return jsonify({
            'success': True,
            'total_reviews': total_reviews,
            'average_rating': average_rating
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'خطا در دریافت آمار'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Reviews Backend',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🌟 Reviews Backend Server Starting...")
    print("📝 Endpoint: http://localhost:5004")
    print("📊 Submit Review: POST /api/reviews")
    print("�  Get Stats: GET /api/reviews/stats")
    print("💾 Reviews saved to: customer_reviews.csv & customer_reviews.txt")
    
    app.run(host='0.0.0.0', port=5004, debug=True)
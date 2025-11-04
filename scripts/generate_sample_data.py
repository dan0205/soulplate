"""
샘플 Yelp 데이터 생성 스크립트
실제 Yelp 데이터가 없을 때 데모용으로 사용
"""

import json
import random
from datetime import datetime, timedelta
import os

# 설정
NUM_USERS = 1000
NUM_BUSINESSES = 500
NUM_REVIEWS = 5000

# 샘플 데이터
CITIES = ["Las Vegas", "Phoenix", "Charlotte", "Scottsdale", "Pittsburgh", "Toronto", "Montreal"]
STATES = ["NV", "AZ", "NC", "PA", "ON", "QC"]
CATEGORIES = [
    "Restaurants", "Shopping", "Food", "Beauty & Spas", "Health & Medical",
    "Home Services", "Nightlife", "Bars", "Event Planning & Services",
    "Arts & Entertainment", "Hotels & Travel", "Active Life",
    "Automotive", "Local Services", "Mass Media", "Pets",
    "Professional Services", "Public Services & Government", "Religious Organizations"
]

FIRST_NAMES = ["John", "Jane", "Michael", "Emily", "David", "Sarah", "Chris", "Lisa", 
               "Daniel", "Jessica", "James", "Ashley", "Robert", "Amanda", "William"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
              "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson"]

BUSINESS_TYPES = ["Restaurant", "Cafe", "Bar", "Salon", "Spa", "Gym", "Store", 
                  "Hotel", "Theater", "Museum", "Park"]

REVIEW_TEMPLATES = [
    "Great {aspect}! Would definitely recommend.",
    "The {aspect} was amazing. {detail}",
    "Not bad, but the {aspect} could be better.",
    "Terrible {aspect}. Very disappointed.",
    "Outstanding {aspect}! Will come back again.",
    "Average {aspect}. Nothing special.",
    "Loved the {aspect}! {detail}",
    "The {aspect} needs improvement.",
]

ASPECTS = ["service", "food", "atmosphere", "location", "quality", "value", "experience"]
DETAILS = [
    "Staff was very friendly.",
    "Wait time was reasonable.",
    "Place was very clean.",
    "Prices are fair.",
    "Highly recommend this place!",
    "Will definitely return.",
]

def generate_user_id():
    """Generate random user ID"""
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=22))

def generate_business_id():
    """Generate random business ID"""
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=22))

def generate_users():
    """Generate sample user data"""
    print(f"Generating {NUM_USERS} users...")
    users = []
    
    for i in range(NUM_USERS):
        user = {
            "user_id": generate_user_id(),
            "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "review_count": random.randint(1, 100),
            "yelping_since": f"{random.randint(2008, 2022)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "useful": random.randint(0, 50),
            "funny": random.randint(0, 30),
            "cool": random.randint(0, 40),
            "fans": random.randint(0, 20),
            "average_stars": round(random.uniform(2.0, 5.0), 1),
            "compliment_hot": random.randint(0, 10),
            "compliment_more": random.randint(0, 10),
            "compliment_profile": random.randint(0, 10),
            "compliment_cute": random.randint(0, 10),
            "compliment_list": random.randint(0, 10),
            "compliment_note": random.randint(0, 10),
            "compliment_plain": random.randint(0, 10),
            "compliment_cool": random.randint(0, 10),
            "compliment_funny": random.randint(0, 10),
            "compliment_writer": random.randint(0, 10),
            "compliment_photos": random.randint(0, 10),
            "elite": "",
            "friends": ""
        }
        users.append(user)
    
    return users

def generate_businesses():
    """Generate sample business data"""
    print(f"Generating {NUM_BUSINESSES} businesses...")
    businesses = []
    
    for i in range(NUM_BUSINESSES):
        city = random.choice(CITIES)
        state = random.choice(STATES)
        business_type = random.choice(BUSINESS_TYPES)
        
        business = {
            "business_id": generate_business_id(),
            "name": f"{random.choice(FIRST_NAMES)}'s {business_type}",
            "address": f"{random.randint(100, 9999)} Main St",
            "city": city,
            "state": state,
            "postal_code": f"{random.randint(10000, 99999)}",
            "latitude": round(random.uniform(30.0, 45.0), 6),
            "longitude": round(random.uniform(-120.0, -70.0), 6),
            "stars": round(random.uniform(1.0, 5.0), 1),
            "review_count": random.randint(5, 500),
            "is_open": random.choice([0, 1]),
            "attributes": {
                "RestaurantsPriceRange2": str(random.randint(1, 4)),
                "BusinessParking": json.dumps({"garage": random.choice([True, False]), 
                                               "street": random.choice([True, False])}),
                "BikeParking": random.choice([True, False]),
                "RestaurantsDelivery": random.choice([True, False]),
                "RestaurantsTakeOut": random.choice([True, False])
            },
            "categories": ", ".join(random.sample(CATEGORIES, k=random.randint(1, 3))),
            "hours": {
                "Monday": "9:0-22:0",
                "Tuesday": "9:0-22:0",
                "Wednesday": "9:0-22:0",
                "Thursday": "9:0-22:0",
                "Friday": "9:0-23:0",
                "Saturday": "10:0-23:0",
                "Sunday": "10:0-20:0"
            }
        }
        businesses.append(business)
    
    return businesses

def generate_reviews(users, businesses):
    """Generate sample review data"""
    print(f"Generating {NUM_REVIEWS} reviews...")
    reviews = []
    
    for i in range(NUM_REVIEWS):
        user = random.choice(users)
        business = random.choice(businesses)
        stars = random.randint(1, 5)
        
        # Generate review text based on stars
        aspect = random.choice(ASPECTS)
        detail = random.choice(DETAILS) if stars >= 4 else ""
        template = random.choice(REVIEW_TEMPLATES)
        text = template.format(aspect=aspect, detail=detail).strip()
        
        # Generate random date within last 2 years
        days_ago = random.randint(0, 730)
        review_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        
        review = {
            "review_id": ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=22)),
            "user_id": user["user_id"],
            "business_id": business["business_id"],
            "stars": stars,
            "useful": random.randint(0, 20),
            "funny": random.randint(0, 10),
            "cool": random.randint(0, 15),
            "text": text,
            "date": review_date
        }
        reviews.append(review)
    
    return reviews

def save_json(data, filename):
    """Save data to JSON file (newline-delimited JSON format like Yelp)"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data)} records to {filename}")

def main():
    """Main function to generate all sample data"""
    print("=" * 50)
    print("Generating Sample Yelp Dataset")
    print("=" * 50)
    
    # Generate data
    users = generate_users()
    businesses = generate_businesses()
    reviews = generate_reviews(users, businesses)
    
    # Save to files
    save_json(users, "data/raw/yelp_academic_dataset_user.json")
    save_json(businesses, "data/raw/yelp_academic_dataset_business.json")
    save_json(reviews, "data/raw/yelp_academic_dataset_review.json")
    
    print("\n" + "=" * 50)
    print("Sample data generation completed!")
    print("=" * 50)
    print(f"Users: {len(users)}")
    print(f"Businesses: {len(businesses)}")
    print(f"Reviews: {len(reviews)}")
    print("\nFiles saved to data/raw/")

if __name__ == "__main__":
    main()


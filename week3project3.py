# AI Recommendation System
# Project 3 - DecodeLabs

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset (items to recommend)
items = {
    "Python Programming Course":
        "python coding programming software development",

    "Machine Learning Basics":
        "artificial intelligence machine learning data science",

    "Web Development Bootcamp":
        "html css javascript web frontend backend",

    "Graphic Design Masterclass":
        "design photoshop illustrator creativity graphics",

    "Cybersecurity Fundamentals":
        "security hacking networks cybersecurity protection",

    "Data Science Workshop":
        "data analytics machine learning statistics python"
}

# Convert item descriptions into lists
titles = list(items.keys())
descriptions = list(items.values())

# User input
user_interest = input("Enter your interests: ").lower()

# Combine dataset + user input
all_text = descriptions + [user_interest]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)

# Similarity Calculation
user_vector = tfidf_matrix[-1]
item_vectors = tfidf_matrix[:-1]

similarities = cosine_similarity(user_vector, item_vectors)

# Find best match
best_match_index = similarities.argmax()
best_score = similarities[0][best_match_index]

# Output recommendation
print("\nRecommended Item:")
print(titles[best_match_index])

print("Similarity Score:", round(best_score, 2))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime, timedelta
import functools


PRODUCTS_FILE_NAME = 'products.csv'
USERS_FILE_NAME = 'users.csv'
INTERACTIONS_FILE_NAME = 'interactions.csv'
SYSTEM_LOG_FILE = 'system_log.txt'


def log_system_event(message_content):
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = open(SYSTEM_LOG_FILE, 'a')
    log_file.write(f"[{current_timestamp}] {message_content}\n")
    log_file.close()


def generate_dummy_data_files():

    if not os.path.exists(PRODUCTS_FILE_NAME):
        products_data_dictionary = {
            'ProductID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'ProductName': ['Laptop', 'Smartphone', 'Headphones', 'Monitor', 'Keyboard', 'Mouse', 'Smartwatch', 'Tablet', 'Charger', 'Webcam'],
            'Category': ['Electronics', 'Electronics', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics', 'Accessories', 'Accessories'],
            'Price': [108000, 72000, 13500, 27000, 9000, 4500, 22500, 54000, 2700, 7200],
            'Rating': [4.5, 4.7, 4.2, 4.6, 4.1, 4.0, 4.3, 4.4, 4.8, 3.9]
        }
        products_dataframe = pd.DataFrame(products_data_dictionary)
        products_dataframe.to_csv(PRODUCTS_FILE_NAME, index=False)
        log_system_event("Created products.csv")


    if not os.path.exists(USERS_FILE_NAME):
        users_data_dictionary = {
            'UserID': [101, 102, 103, 104, 105],
            'Name': ['Sairaj', 'Omkar', 'Sarthak', 'Danish', 'Jaikishan'],
            'Age': [18, 28, 38, 19, 49],
            'Location': ['Kharghar', 'Ghatkopar', 'Ghatkopar', 'Govandi', 'Khandeshwar']
        }
        users_dataframe = pd.DataFrame(users_data_dictionary)
        users_dataframe.to_csv(USERS_FILE_NAME, index=False)
        log_system_event("Created users.csv")


    if not os.path.exists(INTERACTIONS_FILE_NAME):
        possible_actions = ['View', 'Add-to-Cart', 'Purchase']

        user_ids_list = []
        product_ids_list = []
        actions_list = []
        timestamps_list = []

        for i in range(50):
            user_ids_list.append(random.choice([101, 102, 103, 104, 105]))
            product_ids_list.append(random.choice(range(101, 111)))
            actions_list.append(random.choice(possible_actions))
            timestamps_list.append((datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d %H:%M:%S"))

        interactions_data_dictionary = {
            'UserID': user_ids_list,
            'ProductID': product_ids_list,
            'Action': actions_list,
            'Timestamp': timestamps_list
        }
        interactions_dataframe = pd.DataFrame(interactions_data_dictionary)
        interactions_dataframe.to_csv(INTERACTIONS_FILE_NAME, index=False)
        log_system_event("Created interactions.csv")


def notify_user_status(function_to_decorate):
    def wrapper_function(*args, **kwargs):
        print("Running " + function_to_decorate.__name__ + "...")
        log_system_event("Started " + function_to_decorate.__name__)

        result_of_function = function_to_decorate(*args, **kwargs)

        print("Finished " + function_to_decorate.__name__)
        log_system_event("Completed " + function_to_decorate.__name__)
        return result_of_function
    return wrapper_function


class RecommendationEngine:
    def __init__(self):
        self.products_dataframe = None
        self.users_dataframe = None
        self.interactions_dataframe = None
        self.merged_dataframe = None


    @notify_user_status
    def load_data_from_files(self):
        self.products_dataframe = pd.read_csv(PRODUCTS_FILE_NAME)
        self.users_dataframe = pd.read_csv(USERS_FILE_NAME)
        self.interactions_dataframe = pd.read_csv(INTERACTIONS_FILE_NAME)


    @notify_user_status
    def preprocess_all_data(self):
        self.interactions_dataframe['Timestamp'] = pd.to_datetime(self.interactions_dataframe['Timestamp'])

        weight_mapping_dictionary = {'View': 1, 'Add-to-Cart': 2, 'Purchase': 3}
        self.interactions_dataframe['Weight'] = self.interactions_dataframe['Action'].map(weight_mapping_dictionary)

        self.merged_dataframe = pd.merge(self.interactions_dataframe, self.products_dataframe, on='ProductID', how='left')

        price_band_function = lambda price: "Budget" if price < 45000 else "Premium"
        self.products_dataframe['PriceBand'] = self.products_dataframe['Price'].apply(price_band_function)

        self.merged_dataframe = self.merged_dataframe.fillna(0)


    @notify_user_status
    def calculate_popularity_of_products(self):
        popularity_ranking = self.interactions_dataframe.groupby('ProductID')['Weight'].sum().reset_index()
        popularity_ranking = pd.merge(popularity_ranking, self.products_dataframe[['ProductID', 'ProductName', 'Price']], on='ProductID')
        return popularity_ranking.sort_values(by='Weight', ascending=False)


    def recommend_popular_items_list(self):
        top_popular_items = self.calculate_popularity_of_products()
        return top_popular_items.head(10)


    @notify_user_status
    def get_similar_users_list(self, target_user_id):
        if target_user_id not in self.users_dataframe['UserID'].values:
            return []

        user_product_pivot_table = self.interactions_dataframe.pivot_table(index='UserID', columns='ProductID', values='Weight', fill_value=0)

        if target_user_id not in user_product_pivot_table.index:
            return []

        target_user_vector_data = user_product_pivot_table.loc[target_user_id]
        similarity_scores = user_product_pivot_table.corrwith(target_user_vector_data, axis=1)
        similarity_scores = similarity_scores.sort_values(ascending=False)

        list_of_users = similarity_scores.index.tolist()
        final_similar_users = []

        for user in list_of_users:
            if user != target_user_id:
                final_similar_users.append(user)

        return final_similar_users[:5]


    @notify_user_status
    def content_based_recommendation_logic(self, product_id_to_check):
        if product_id_to_check not in self.products_dataframe['ProductID'].values:
            return pd.DataFrame()

        target_product_details = self.products_dataframe[self.products_dataframe['ProductID'] == product_id_to_check].iloc[0]

        recommended_products = self.products_dataframe[
            (self.products_dataframe['Category'] == target_product_details['Category']) &
            (self.products_dataframe['ProductID'] != product_id_to_check)
        ].copy()

        recommended_products['Reason'] = "Similar to " + target_product_details['ProductName']
        return recommended_products[['ProductName', 'Reason', 'Price', 'Rating']]


    @notify_user_status
    def recommend_for_specific_user(self, user_id_input):
        recommendation_list = []

        user_interaction_history = self.interactions_dataframe[self.interactions_dataframe['UserID'] == user_id_input].sort_values(by='Timestamp', ascending=False)

        if len(user_interaction_history) > 0:
            last_viewed_product_id = user_interaction_history.iloc[0]['ProductID']

            content_based_suggestions = self.content_based_recommendation_logic(last_viewed_product_id)
            for index, row in content_based_suggestions.iterrows():
                recommendation_list.append(row['ProductName'] + " - Similar to last viewed")

        popular_items_suggestions = self.recommend_popular_items_list()
        for index, row in popular_items_suggestions.head(3).iterrows():
            recommendation_list.append(row['ProductName'] + " - Trending/Popular")

        final_recommendation_list = []
        for item in recommendation_list:
            if item not in final_recommendation_list:
                final_recommendation_list.append(item)

        return final_recommendation_list[:10]


    @notify_user_status
    def save_final_recommendations(self, user_id_input, list_of_recommendations):
        file_name_for_save = "recommendations_user_" + str(user_id_input) + ".csv"
        dataframe_to_save = pd.DataFrame(list_of_recommendations, columns=['Recommendation'])
        dataframe_to_save.to_csv(file_name_for_save, index=False)
        print("Saved to " + file_name_for_save)
        log_system_event("Saved recommendations to " + file_name_for_save)


    @notify_user_status
    def visualize_usage_trends(self):
        popularity_data = self.calculate_popularity_of_products().head(10)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(popularity_data['ProductName'], popularity_data['Price'], color='skyblue')
        plt.title('Price of Top 10 Trending Products')
        plt.xlabel('Product')
        plt.ylabel('Price (INR)')
        plt.xticks(rotation=45)

        category_distribution = self.products_dataframe['Category'].value_counts()

        plt.subplot(1, 2, 2)
        plt.pie(category_distribution, labels=category_distribution.index, autopct='%1.1f%%', startangle=140)
        plt.title('Category Distribution')

        plt.tight_layout()
        plt.show()


generate_dummy_data_files()

recommendation_engine_instance = RecommendationEngine()
recommendation_engine_instance.load_data_from_files()
recommendation_engine_instance.preprocess_all_data()
recommendation_engine_instance.calculate_popularity_of_products()


print("Available User IDs: " + str(recommendation_engine_instance.users_dataframe['UserID'].tolist()))
target_user_id_variable = int(input("Enter User ID to generate recommendations for: "))

if target_user_id_variable not in recommendation_engine_instance.users_dataframe['UserID'].values:
    print("User ID not found! Defaulting to 101.")
    target_user_id_variable = 101

print("Generatng recommendations for User " + str(target_user_id_variable))

recommendations_result = recommendation_engine_instance.recommend_for_specific_user(target_user_id_variable)

print("Recommendations for User " + str(target_user_id_variable) + ":")
counter_variable = 1
for recommendation_item in recommendations_result:
    print(str(counter_variable) + ". " + recommendation_item)
    counter_variable = counter_variable + 1

recommendation_engine_instance.save_final_recommendations(target_user_id_variable, recommendations_result)
recommendation_engine_instance.visualize_usage_trends()
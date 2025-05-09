import random
from faker import Faker
import pandas as pd

# Initialize Faker for generating synthetic names and text
fake = Faker()

# Function to generate a synthetic historical-style speech
def generate_speech():
    return " ".join(fake.paragraphs(nb=random.randint(3, 6)))

# Generate 1000 synthetic rows
synthetic_data = {
    "speaker": [fake.name() for _ in range(10000)],
    "date": [fake.date_between(start_date='-100y', end_date='today').strftime("%Y-%m-%d") for _ in range(10000)],
    "text": [generate_speech() for _ in range(10000)]
}

# Create DataFrame from synthetic data
df_synthetic = pd.DataFrame(synthetic_data)

df = pd.read_csv('historical_speeches.csv')
# Concatenate with the original DataFrame
df_combined = pd.concat([df, df_synthetic], ignore_index=True)

# Save to new CSV
output_path = "historical_speeches_expanded.csv"
df_combined.to_csv(output_path, index=False)

output_path

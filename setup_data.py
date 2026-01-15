import pandas as pd
import numpy as np
import random

def setup_data():
    print("⏳ Generating Sent-X Dataset with Hashtags...")
    data = {"tweet": [], "sentiment": [], "followers": [], "following": [], "id": [], "topic": []}
    
    topics = ['Apple', 'Samsung', 'Crypto', 'Politics', 'Tesla', 'Climate']
    
    # Templates NOW INCLUDE HASHTAGS
    templates = {
        "happy": [
            "I love this update! #Amazing #Tech", 
            "Best purchase ever. #Recommended #LoveIt", 
            "Finally some good news! #Winning #Happy",
            "This feature is a game changer. #Innovation #Future"
        ],
        "angry": [
            "This is terrible service. #Fail #Angry", 
            "Worst experience ever. #Scam #Refund", 
            "I hate this bug! #Broken #FixIt",
            "Unacceptable behavior. #Boycott #Mad"
        ],
        "fear": [
            "Market is crashing! #Panic #Crash", 
            "Scary news coming out. #Fear #Alert", 
            "Is my data safe? #Privacy #Hack",
            "I am worried about the future. #Uncertainty #Risk"
        ],
        "neutral": [
            "Just checking in. #Update #News", 
            "Interesting article. #Read #Info", 
            "What do you think? #Question #Poll",
            "Link in bio. #Link #Profile"
        ]
    }
    
    for i in range(2500):
        topic = random.choice(topics)
        sent = random.choice(list(templates.keys()))
        base_tweet = random.choice(templates[sent])
        
        # Add Topic Hashtag (e.g., #Apple)
        tweet = f"{base_tweet} #{topic} {random.randint(1,999)}"
        
        data["tweet"].append(tweet)
        data["sentiment"].append(sent)
        data["id"].append(i)
        data["topic"].append(topic)
        
        # Inject Bots (30%)
        if random.random() < 0.3:
            data["followers"].append(random.randint(0, 50))
            data["following"].append(random.randint(1000, 5000))
            if random.random() > 0.5: data["sentiment"][-1] = "angry"
        else:
            data["followers"].append(random.randint(100, 2000))
            data["following"].append(random.randint(100, 2000))

    df = pd.DataFrame(data)
    df.to_csv("twitter_sentiment_dataset.csv", index=False)
    print("✅ Data Ready with Hashtags! Run 'streamlit run app_major.py'")

if __name__ == "__main__":
    setup_data()
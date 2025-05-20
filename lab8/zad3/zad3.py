import praw


reddit = praw.Reddit(
    client_id="w1cvkxyeB_Vsomo67BpTGg",
    client_secret="LvyP0piSQNTsy9cA10TRK31GZOxLsg",
    user_agent="Sad_Objective_8497"
)


posts = []
subreddit = reddit.subreddit("ArtificialInteligence") 

for post in subreddit.hot(limit=100): 
    posts.append({
        "title": post.title,
        "text": post.selftext,
        "url": post.url
    })


import json
with open("reddit_posts.json", "w", encoding="utf-8") as f:
    json.dump(posts, f, ensure_ascii=False, indent=2)

print("Zapisano 100 postów do pliku reddit_posts.json")

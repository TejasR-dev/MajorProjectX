import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import time
import random
from fpdf import FPDF
import base64
import re
import numpy as np
import requests
from datetime import datetime
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie


# ================= 1. CONFIGURATION =================
st.set_page_config(
    page_title="Sent-X: Social Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed" # Starts cleaner
)

# Constants
DATA_FILE = "twitter_sentiment_dataset.csv"
MODEL_FILE = "sentiment_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
METRICS_FILE = "metrics.csv"

try:
    GOOGLE_API_KEY = st.secrets["Your_actual_api_key"]
    GOOGLE_CSE_ID = st.secrets["Your_actual_cse_id"]
except Exception:
    GOOGLE_API_KEY = ""
    GOOGLE_CSE_ID = ""

# ================= 2. PROFESSIONAL STYLING (Glassmorphism) =================

# Load Lottie Animation from URL
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# Animation Assets (DEFINED HERE TO FIX YOUR ERROR)
lottie_brain = load_lottieurl("https://lottie.host/5a093758-c076-4649-8c97-6a5605d3b379/1T8J0s3o7s.json") # AI Brain
lottie_search = load_lottieurl("https://lottie.host/9e473262-4363-4700-9285-4876798889d1/F5q7v7y2qR.json") # Scanning
lottie_bot = load_lottieurl("https://lottie.host/80562479-0524-4f49-b0c6-339275133d1c/D5b2X7v2qR.json") # Robot
lottie_success = load_lottieurl("https://lottie.host/2e23297a-9772-4680-a92c-09757f123405/7Q2Z9v2qR.json") # Checkmark

st.markdown("""
    <style>
    /* Main Background - Dark Cyberpunk Theme */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Navigation Bar Styling */
    .nav-link-selected {
        background-color: #FF4B4B !important;
    }
    
    /* Metric Cards (Glass Effect) */
    /* div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    } */
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
        background-color: #FF4B4B;
        border: none;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.4);
    }
    
    /* Clean up Streamlit UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #E6E6E6;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= 3. UTILITIES & LOGIC =================

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer
    except:
        return None, None

def update_metrics(acc, f1):
    new_entry = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M"), acc, f1]], 
                             columns=["timestamp", "accuracy", "f1_score"])
    try:
        log = pd.read_csv(METRICS_FILE)
        log = pd.concat([log, new_entry], ignore_index=True)
    except:
        log = new_entry
    log.to_csv(METRICS_FILE, index=False)

def train_model_logic():
    try:
        df = pd.read_csv(DATA_FILE).dropna(subset=['tweet', 'sentiment'])
        X = df['tweet']; y = df['sentiment']
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_vec = vectorizer.fit_transform(X)
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_vec, y)
        y_pred = model.predict(X_vec)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        joblib.dump(model, MODEL_FILE)
        joblib.dump(vectorizer, VECTORIZER_FILE)
        update_metrics(acc, f1)
        return model, vectorizer, f"Training Success! Accuracy: {acc:.2%}"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def explain_prediction(model, vectorizer, text):
    try:
        vec = vectorizer.transform([text])
        pred_class = model.predict(vec)[0]
        class_idx = list(model.classes_).index(pred_class)
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[class_idx]
        words = text.lower().split()
        impact = {}
        for word in words:
            if word in vectorizer.vocabulary_:
                idx = vectorizer.vocabulary_[word]
                impact[word] = coefs[idx]
        return sorted(impact.items(), key=lambda x: x[1], reverse=True)
    except:
        return []

def classify_account(followers, following):
    try:
        followers = int(followers); following = int(following)
        if following > 50 and followers < 10: return "Bot"
        if following > (followers * 50): return "Bot"
        return "Human"
    except:
        return "Human"

def analyze_aspects(tweets):
    aspects = {"Battery": 0, "Price": 0, "Service": 0, "Design": 0}
    for tweet in tweets:
        t = str(tweet).lower()
        if any(x in t for x in ["battery", "charge", "power"]): aspects["Battery"] += 1
        if any(x in t for x in ["price", "cost", "expensive"]): aspects["Price"] += 1
        if any(x in t for x in ["service", "support", "help"]): aspects["Service"] += 1
        if any(x in t for x in ["design", "look", "screen"]): aspects["Design"] += 1
    return aspects

def generate_pdf_report(brand, stats, verdict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Sent-X Intelligence Report: {brand}", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Strategic Verdict: {verdict}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Sentiment Breakdown:", ln=True)
    for k, v in stats.items():
        pdf.cell(200, 10, txt=f"- {k}: {v}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

def authenticity_score(survey):
    score = 0
    reasons = []
    if survey["seen_similar"] == "Many times": score += 30; reasons.append("Repeated Content (Spam Risk)")
    if survey["sentiment_extreme"]: score += 20; reasons.append("Inflammatory Language")
    if survey["account_flagged"]: score += 30; reasons.append("User Reported Previously")
    if survey["brand_related"]: score += 20; reasons.append("Promotional/Brand Messaging")
    
    if score >= 70: verdict = "High Risk: Coordinated Campaign"
    elif score >= 40: verdict = "Medium Risk: Suspicious Activity"
    else: verdict = "Low Risk: Likely Genuine"
    return score, verdict, reasons

def google_event_context(query):
    # Simulation Mode
    if "YOUR_" in GOOGLE_API_KEY:
        time.sleep(1)
        return [
            f"SIMULATION: Trending news about '{query}' detected.",
            "SIMULATION: Market analyst reports volatility in this sector.",
            "Note: Set API keys in code for live results."
        ]
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": 3}
        r = requests.get(url, params=params)
        data = r.json()
        summaries = [item["snippet"] for item in data.get("items", [])]
        return summaries if summaries else ["No public events found."]
    except Exception as e:
        return [f"Context fetch failed: {e}"]

import urllib.parse

import urllib.parse

def generate_search_links(subject):
    q = urllib.parse.quote_plus(subject.strip())

    return {
        "Google Search": f"https://www.google.com/search?q={q}",
        "Google News": f"https://www.google.com/search?q={q}&tbm=nws",
        "Twitter / X": f"https://twitter.com/search?q={q}",
        "YouTube": f"https://www.youtube.com/results?search_query={q}"
    }



def save_feedback(tweet, correct_label):
    try:
        df = pd.read_csv(DATA_FILE)
        new_row = pd.DataFrame([{
            "tweet": tweet, "sentiment": correct_label, "id": len(df)+1, "topic": "User_Feedback",
            "followers": random.randint(100,500), "following": random.randint(100,500)
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        return True
    except: return False

def extract_hashtags(text):
    return re.findall(r"#\w+", text)

# ================= 4. NAVIGATION =================
page = option_menu(
    menu_title="Sent-X Intelligence",
    options=["📊 Sentiment Dashboard", "⚔️ Brand Battle", "🤖 Bot Forensics", "🧠 Context Intel", "⚙️ Settings"],
    icons=['bar-chart-line', 'diagram-2', 'robot', 'search', 'gear'],
    menu_icon="cpu", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "5px", "background-color": "#0E1117"},
        "icon": {"color": "#FAFAFA", "font-size": "16px"}, 
        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "color": "#FAFAFA"},
        "nav-link-selected": {"background-color": "#FF4B4B", "color": "white"},
    }
)

model, vectorizer = load_resources()

# ================= PAGE 1: DASHBOARD =================
if page == "📊 Sentiment Dashboard":
    st.title("📊 Real-Time Sentiment Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_area("Enter Tweet / Text", height=100)
        if st.button("Analyze"):
            if user_input and model:
                st.session_state["analyzed_tweet"] = user_input 
                vec = vectorizer.transform([user_input])
                st.session_state["dash_pred"] = model.predict(vec)[0]
                st.session_state["dash_conf"] = model.predict_proba(vec).max()

    if "dash_pred" in st.session_state:
        sentiment = st.session_state["dash_pred"]
        conf = st.session_state["dash_conf"]
        
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Sentiment", sentiment.upper())
        m2.metric("Confidence", f"{conf:.1%}")
        
        with m3:
            st.write("**Virality Risk**")
            if sentiment in ['angry', 'fear']: st.progress(0.9, "High Risk")
            elif sentiment == 'happy': st.progress(0.6, "Moderate")
            else: st.progress(0.2, "Low")

        st.markdown("### 🧐 Explainability (XAI)")
        exp = explain_prediction(model, vectorizer, st.session_state.get("analyzed_tweet", ""))
        if exp:
            df_exp = pd.DataFrame(exp, columns=["Word", "Impact"])
            df_exp["Type"] = df_exp["Impact"].apply(lambda x: "Positive" if x>0 else "Negative")
            st.plotly_chart(px.bar(df_exp, x="Impact", y="Word", color="Type", orientation='h', color_discrete_map={"Positive":"#2ecc71","Negative":"#e74c3c"}), use_container_width=True)

        with st.expander("📝 Teach the AI (Feedback Loop)"):
            correct = st.selectbox("Correct Label", ["happy", "angry", "fear", "sad", "supportive", "confusion"])
            if st.button("Submit Correction"):
                if save_feedback(st.session_state.get("analyzed_tweet", ""), correct):
                    st.success("Feedback Saved! Retrain in Settings.")

    st.markdown("---")
    st.subheader("📈 Knowledge Base Overview")
    try:
        df = pd.read_csv(DATA_FILE)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(df, names='sentiment', title='Overall Sentiment Distribution', hole=0.4), use_container_width=True)
        c2.plotly_chart(px.bar(df['topic'].value_counts().head(10), orientation='h', title="Top Discussed Topics"), use_container_width=True)
    except: st.warning("Dataset not found. Run setup_data.py first.")

# ================= PAGE 2: BRAND BATTLE =================
elif page == "⚔️ Brand Battle":
    st.title("⚔️ Competitive Intelligence")
    c1, c2 = st.columns(2)
    b1 = c1.text_input("Brand A", "Apple"); b2 = c2.text_input("Brand B", "Samsung")
    
    if st.button("Simulate Battle"):
        try:
            df = pd.read_csv(DATA_FILE)
            sub_a = df.sample(n=min(len(df), 300)); sub_b = df.sample(n=min(len(df), 300))
            
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(sub_a, names='sentiment', title=f"{b1} Sentiment", hole=0.4), use_container_width=True)
            c2.plotly_chart(px.pie(sub_b, names='sentiment', title=f"{b2} Sentiment", hole=0.4), use_container_width=True)
            
            st.subheader("Deep Dive: Topic Extraction")
            aspects_a = analyze_aspects(sub_a['tweet']); aspects_b = analyze_aspects(sub_b['tweet'])
            df_asp = pd.DataFrame([aspects_a, aspects_b], index=[b1, b2]).T.reset_index().melt(id_vars="index", var_name="Brand", value_name="Mentions")
            st.plotly_chart(px.bar(df_asp, x="index", y="Mentions", color="Brand", barmode="group"), use_container_width=True)

            pos_a = len(sub_a[sub_a['sentiment'].isin(['happy','supportive'])])
            neg_a = len(sub_a[sub_a['sentiment'].isin(['angry','fear','sad'])])
            verdict = "BUY / BULLISH 📈" if (pos_a - neg_a) > 0 else "SELL / BEARISH 📉"
            st.metric(f"Verdict for {b1}", verdict, delta=int(pos_a - neg_a))
            
            pdf = generate_pdf_report(b1, sub_a['sentiment'].value_counts().to_dict(), verdict)
            b64 = base64.b64encode(pdf).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Report.pdf" style="padding:10px; background-color:#FF4B4B; color:white; border-radius:5px; text-decoration:none;">📄 Download Executive Report</a>', unsafe_allow_html=True)
        except Exception as e: st.error(f"Error: {e}")

# ================= PAGE 3: BOT FORENSICS =================
elif page == "🤖 Bot Forensics":
    st.title("🤖 Authenticity & Forensics")
    tab1, tab2 = st.tabs(["📊 Dataset Forensics (Automated)", "🕵️ Single Tweet Check (Manual)"])
    
    with tab1:
        st.markdown("### 📡 Network Topology Analysis")
        try:
            df = pd.read_csv(DATA_FILE)
            if 'followers' not in df.columns: st.error("Run setup_data.py!")
            else:
                df['User Type'] = df.apply(lambda x: classify_account(x['followers'], x['following']), axis=1)
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.pie(df, names='User Type', title="Bot Prevalence"), use_container_width=True)
                sent_counts = df.groupby(['User Type', 'sentiment']).size().reset_index(name='Count')
                c2.plotly_chart(px.bar(sent_counts, x="sentiment", y="Count", color="User Type", barmode="group"), use_container_width=True)
                
                bots = df[df['User Type'] == "Bot"]
                bot_neg = len(bots[bots['sentiment'].isin(['angry', 'fear'])]) / len(bots) * 100 if len(bots) > 0 else 0
                st.metric("Bot Negativity Rate", f"{bot_neg:.1f}%")
                if bot_neg > 40: st.error("🚨 ALERT: Bots are amplifying negativity!")
        except Exception as e: st.error(f"Error: {e}")

    with tab2:
        st.markdown("### 🕵️ Manual Authenticity Engine")
        target_tweet = st.session_state.get("analyzed_tweet", "")
        if not target_tweet: st.warning("Analyze a tweet in Dashboard first.")
        else:
            st.info(f"Analyzing: {target_tweet}")
            c1, c2 = st.columns(2)
            seen_similar = c1.selectbox("Seen similar?", ["No", "A few times", "Many times"])
            category = c2.selectbox("Topic", ["Brand", "Politics", "News", "Personal"])
            c3, c4 = st.columns(2)
            sentiment_extreme = c3.checkbox("Extreme language?")
            account_flagged = c4.checkbox("Suspicious profile?")
            
            if st.button("Run Forensic Check"):
                score, verdict, reasons = authenticity_score({
                    "seen_similar": seen_similar, "sentiment_extreme": sentiment_extreme,
                    "account_flagged": account_flagged, "brand_related": category == "Brand"
                })
                st.metric("Risk Score", f"{score}/100")
                if score > 60: st.error(f"VERDICT: {verdict}")
                else: st.success(f"VERDICT: {verdict}")
                for r in reasons: st.write(f"❌ {r}")

# ================= PAGE 4: CONTEXT INTELLIGENCE (UPGRADED) =================
elif page == "🧠 Context Intel":
    st.markdown("<h1 style='text-align: center;'>🧠 Context & Hashtag Intelligence</h1>", unsafe_allow_html=True)
    
    target_tweet = st.session_state.get("analyzed_tweet", "")
    
    if not target_tweet: 
        st.warning("Please analyze a tweet in the Dashboard first.")
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write(f"**Target Tweet:** \"{target_tweet}\"")
        
        # 1. EXTRACT HASHTAGS
        tags = extract_hashtags(target_tweet)
        if tags:
            st.success(f"**Detected Entities:** {', '.join(tags)}")
        else:
            st.info("No hashtags detected in this tweet.")
        st.markdown('</div>', unsafe_allow_html=True)

        # 2. HASHTAG INTELLIGENCE (The Upgrade)
        if tags:
            st.subheader("📊 Hashtag Deep Dive")
            
            try:
                df = pd.read_csv(DATA_FILE)
                
                # Loop through each tag found
                for tag in tags:
                    with st.expander(f"Analysis for {tag}", expanded=True):
                        # Filter data for this tag
                        tag_data = df[df['tweet'].str.contains(tag, case=False, na=False)]
                        
                        if len(tag_data) > 0:
                            c1, c2, c3 = st.columns([1, 1, 2])
                            
                            # Metric: Volume
                            c1.metric(f"Total Mentions", len(tag_data))
                            
                            # Metric: Dominant Sentiment
                            top_sent = tag_data['sentiment'].mode()[0]
                            c2.metric(f"Dominant Emotion", top_sent.upper())
                            
                            # Chart: Sentiment Distribution for this Tag
                            fig = px.pie(tag_data, names='sentiment', title=f"Sentiment Profile: {tag}", hole=0.5)
                            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=0))
                            c3.plotly_chart(fig, use_container_width=True)
                            
                            # Similar Tweets
                            st.caption(f"Recent Tweets using {tag}:")
                            for t in tag_data['tweet'].sample(min(3, len(tag_data))).tolist():
                                st.text(f"• {t}")
                        else:
                            st.warning(f"No historical data found for {tag} in our database.")
            except Exception as e:
                st.error(f"Database Error: {e}")

        # 3. OSINT CONTEXT (Updated to use hashtags)
        st.markdown("---")
        st.subheader("🌍 Real-World Context (OSINT)")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # Auto-fill subject using the first hashtag if available
            default_subject = tags[0].replace("#", "") if tags else ""
            subject = st.text_input("Subject / Query", value=default_subject, placeholder="e.g. Apple")
            # date_seen = st.date_input("Date", datetime.now())
            
            if st.button("🔍 Search on Web"):
                if not subject.strip():
                    st.warning("Please enter a subject.")
                else:
                    links = generate_search_links(subject)

                    st.markdown("### 🔗 Open Trusted Sources")
                    for name, url in links.items():
                        st.markdown(f"- [{name}]({url})")

                    st.caption(
                        "ℹ️ External links open trusted platforms for independent context exploration."
                    )

        
        with col2:
            if lottie_search:
                st_lottie(lottie_search, height=120, key="search_anim")
            else:
                st.info("🔍 Context analysis in progress...")



# ================= PAGE 5: SETTINGS =================
elif page == "⚙️ Settings":
    st.title("⚙️ System Maintenance")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚀 Model Actions")
        if st.button("🔄 Retrain Model"):
            with st.spinner("Training..."):
                m, v, msg = train_model_logic()
                if m: st.success(msg)
                else: st.error(msg)
    with col2:
        st.subheader("📊 Performance History")
        try:
            met = pd.read_csv(METRICS_FILE)
            st.line_chart(met.set_index("timestamp")['accuracy'])
            st.write(f"Current Accuracy: {met['accuracy'].iloc[-1]:.2%}")

        except: st.info("No training history available.")

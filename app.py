from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import google.generativeai as genai
import json
import re
from collections import Counter
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Configure Gemini AI directly with API key
GEMINI_API_KEY = "AIzaSyB5THv6YOEjy01uEyQmfI9T-iTci--11ZI"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SentimentAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    text_content = db.Column(db.Text, nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)  # basic, detailed, psychological, social
    sentiment = db.Column(db.String(20), nullable=False)  # positive, negative, neutral
    confidence_score = db.Column(db.Float, nullable=False)
    emotions = db.Column(db.Text, nullable=True)  # JSON string of emotions
    psychological_insights = db.Column(db.Text, nullable=True)
    keywords = db.Column(db.Text, nullable=True)  # JSON string of keywords
    category = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class BatchAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    batch_name = db.Column(db.String(100), nullable=False)
    total_texts = db.Column(db.Integer, nullable=False)
    overall_sentiment = db.Column(db.String(20), nullable=False)
    summary_insights = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Advanced Sentiment Analysis AI
class SentimentAI:
    def __init__(self):
        self.model = model
    
    def analyze_basic_sentiment(self, text):
        """Basic sentiment analysis with confidence score"""
        prompt = f"""
        Analyze the sentiment of this text and provide a detailed response:
        
        TEXT: {text}
        
        Provide your analysis in this exact JSON format:
        {{
            "sentiment": "positive/negative/neutral",
            "confidence_score": 0.85,
            "explanation": "Brief explanation of why this sentiment was detected",5
            "key_phrases": ["phrase1", "phrase2", "phrase3"],
            "intensity": "low/medium/high"
        }}
        
        Be precise and accurate in your analysis.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text, {
                "sentiment": "neutral",
                "confidence_score": 0.5,
                "explanation": "Unable to determine sentiment",
                "key_phrases": [],
                "intensity": "medium"
            })
        except Exception as e:
            print(f"Error in basic sentiment analysis: {e}")
            return {
                "sentiment": "neutral",
                "confidence_score": 0.5,
                "explanation": f"Analysis error: {str(e)}",
                "key_phrases": [],
                "intensity": "medium"
            }
    
    def analyze_detailed_emotions(self, text):
        """Detailed emotion analysis beyond basic sentiment"""
        prompt = f"""
        Perform a comprehensive emotional analysis of this text:
        
        TEXT: {text}
        
        Provide your analysis in this exact JSON format:
        {{
            "primary_emotion": "joy/anger/sadness/fear/surprise/disgust/trust/anticipation",
            "emotion_scores": {{
                "joy": 0.2,
                "anger": 0.1,
                "sadness": 0.3,
                "fear": 0.1,
                "surprise": 0.0,
                "disgust": 0.0,
                "trust": 0.2,
                "anticipation": 0.1
            }},
            "emotional_intensity": "low/medium/high",
            "mood_indicators": ["indicator1", "indicator2"],
            "emotional_context": "Brief description of emotional context"
        }}
        
        Analyze all emotions present, not just the dominant one.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text, {
                "primary_emotion": "neutral",
                "emotion_scores": {"joy": 0.5, "anger": 0.0, "sadness": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "trust": 0.3, "anticipation": 0.2},
                "emotional_intensity": "medium",
                "mood_indicators": ["neutral tone"],
                "emotional_context": "Neutral emotional state"
            })
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            return {
                "primary_emotion": "neutral",
                "emotion_scores": {"joy": 0.5, "anger": 0.0, "sadness": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "trust": 0.3, "anticipation": 0.2},
                "emotional_intensity": "medium",
                "mood_indicators": [f"Analysis error: {str(e)}"],
                "emotional_context": "Unable to analyze emotions"
            }
    
    def analyze_psychological_insights(self, text):
        """Deep psychological analysis and personality insights"""
        prompt = f"""
        Provide psychological insights based on this text:
        
        TEXT: {text}
        
        Provide your analysis in this exact JSON format:
        {{
            "personality_traits": ["trait1", "trait2", "trait3"],
            "communication_style": "assertive/passive/aggressive/analytical/emotional",
            "stress_indicators": ["indicator1", "indicator2"],
            "confidence_level": "low/medium/high",
            "social_orientation": "introverted/extroverted/ambiverted",
            "decision_making_style": "logical/emotional/balanced",
            "psychological_state": "Brief assessment of mental state",
            "recommendations": ["suggestion1", "suggestion2"]
        }}
        
        Base insights on language patterns, word choice, and expression style.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text, {
                "personality_traits": ["neutral"],
                "communication_style": "balanced",
                "stress_indicators": [],
                "confidence_level": "medium",
                "social_orientation": "ambiverted",
                "decision_making_style": "balanced",
                "psychological_state": "Stable emotional state",
                "recommendations": ["Continue balanced communication"]
            })
        except Exception as e:
            print(f"Error in psychological analysis: {e}")
            return {
                "personality_traits": ["analysis_unavailable"],
                "communication_style": "unknown",
                "stress_indicators": [f"Error: {str(e)}"],
                "confidence_level": "unknown",
                "social_orientation": "unknown",
                "decision_making_style": "unknown",
                "psychological_state": "Unable to analyze",
                "recommendations": ["Please try again"]
            }
    
    def analyze_social_media_sentiment(self, text):
        """Specialized analysis for social media content"""
        prompt = f"""
        Analyze this social media text for sentiment and social indicators:
        
        TEXT: {text}
        
        Provide your analysis in this exact JSON format:
        {{
            "sentiment": "positive/negative/neutral",
            "virality_potential": "low/medium/high",
            "engagement_type": "informative/emotional/controversial/entertaining",
            "social_signals": ["signal1", "signal2"],
            "hashtag_sentiment": "positive/negative/neutral",
            "audience_reaction": "supportive/critical/mixed/indifferent",
            "content_category": "personal/professional/promotional/news",
            "influence_score": 0.75
        }}
        
        Consider social media context, slang, emojis, and informal language.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text, {
                "sentiment": "neutral",
                "virality_potential": "low",
                "engagement_type": "informative",
                "social_signals": ["standard_post"],
                "hashtag_sentiment": "neutral",
                "audience_reaction": "indifferent",
                "content_category": "personal",
                "influence_score": 0.5
            })
        except Exception as e:
            print(f"Error in social media analysis: {e}")
            return {
                "sentiment": "neutral",
                "virality_potential": "low",
                "engagement_type": "informative",
                "social_signals": [f"Error: {str(e)}"],
                "hashtag_sentiment": "neutral",
                "audience_reaction": "indifferent",
                "content_category": "personal",
                "influence_score": 0.5
            }
    
    def analyze_customer_feedback(self, text):
        """Specialized analysis for customer reviews and feedback"""
        prompt = f"""
        Analyze this customer feedback for business insights:
        
        TEXT: {text}
        
        Provide your analysis in this exact JSON format:
        {{
            "satisfaction_level": "very_dissatisfied/dissatisfied/neutral/satisfied/very_satisfied",
            "urgency": "low/medium/high/critical",
            "feedback_category": "product/service/support/delivery/pricing",
            "actionable_items": ["action1", "action2"],
            "customer_intent": "complaint/praise/suggestion/inquiry",
            "business_impact": "low/medium/high",
            "response_priority": "low/medium/high",
            "key_concerns": ["concern1", "concern2"]
        }}
        
        Focus on business-relevant insights and actionable feedback.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text, {
                "satisfaction_level": "neutral",
                "urgency": "medium",
                "feedback_category": "general",
                "actionable_items": ["review_feedback"],
                "customer_intent": "inquiry",
                "business_impact": "medium",
                "response_priority": "medium",
                "key_concerns": ["general_feedback"]
            })
        except Exception as e:
            print(f"Error in customer feedback analysis: {e}")
            return {
                "satisfaction_level": "neutral",
                "urgency": "medium",
                "feedback_category": "general",
                "actionable_items": [f"Error: {str(e)}"],
                "customer_intent": "inquiry",
                "business_impact": "medium",
                "response_priority": "medium",
                "key_concerns": ["analysis_error"]
            }
    
    def analyze_batch_sentiment(self, texts):
        """Analyze multiple texts and provide aggregate insights"""
        if not texts:
            return {"error": "No texts provided"}
        
        # Combine texts for analysis
        combined_text = "\n\n".join(texts[:10])  # Limit to first 10 texts to avoid token limits
        
        prompt = f"""
        Analyze these multiple texts for overall sentiment patterns:
        
        TEXTS:
        {combined_text}
        
        Provide aggregate analysis in this exact JSON format:
        {{
            "overall_sentiment": "positive/negative/neutral",
            "sentiment_distribution": {{
                "positive": 0.4,
                "negative": 0.2,
                "neutral": 0.4
            }},
            "common_themes": ["theme1", "theme2", "theme3"],
            "dominant_emotions": ["emotion1", "emotion2"],
            "trend_analysis": "improving/declining/stable",
            "key_insights": ["insight1", "insight2"],
            "recommendations": ["recommendation1", "recommendation2"]
        }}
        
        Focus on patterns and trends across all texts.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text, {
                "overall_sentiment": "neutral",
                "sentiment_distribution": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "common_themes": ["general_content"],
                "dominant_emotions": ["neutral"],
                "trend_analysis": "stable",
                "key_insights": ["Mixed sentiment patterns"],
                "recommendations": ["Continue monitoring"]
            })
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            return {
                "overall_sentiment": "neutral",
                "sentiment_distribution": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "common_themes": [f"Error: {str(e)}"],
                "dominant_emotions": ["unknown"],
                "trend_analysis": "unknown",
                "key_insights": ["Analysis failed"],
                "recommendations": ["Please try again"]
            }
    
    def _parse_json_response(self, response_text, fallback):
        """Parse JSON response with fallback"""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return fallback
        except:
            return fallback

# Initialize AI Assistant
sentiment_ai = SentimentAI()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered!')
            return redirect(url_for('register'))
        
        user = User(name=name, email=email)
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        session['user_name'] = user.name
        flash('Registration successful!')
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        
        if user:
            session['user_id'] = user.id
            session['user_name'] = user.name
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('User not found!')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get recent analyses
    recent_analyses = SentimentAnalysis.query.filter_by(user_id=session['user_id']).order_by(SentimentAnalysis.created_at.desc()).limit(5).all()
    
    # Get sentiment distribution for charts
    all_analyses = SentimentAnalysis.query.filter_by(user_id=session['user_id']).all()
    sentiment_counts = Counter([analysis.sentiment for analysis in all_analyses])
    
    return render_template('dashboard.html', 
                         analyses=recent_analyses, 
                         sentiment_counts=sentiment_counts)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_text():
    if request.method == 'POST':
        text_content = request.form['text_content']
        analysis_type = request.form.get('analysis_type', 'basic')
        
        # Perform analysis based on type
        if analysis_type == 'basic':
            result = sentiment_ai.analyze_basic_sentiment(text_content)
            emotions = None
            psychological = None
        elif analysis_type == 'detailed':
            basic_result = sentiment_ai.analyze_basic_sentiment(text_content)
            emotion_result = sentiment_ai.analyze_detailed_emotions(text_content)
            result = basic_result
            emotions = emotion_result
            psychological = None
        elif analysis_type == 'psychological':
            basic_result = sentiment_ai.analyze_basic_sentiment(text_content)
            emotion_result = sentiment_ai.analyze_detailed_emotions(text_content)
            psych_result = sentiment_ai.analyze_psychological_insights(text_content)
            result = basic_result
            emotions = emotion_result
            psychological = psych_result
        elif analysis_type == 'social':
            result = sentiment_ai.analyze_social_media_sentiment(text_content)
            emotions = None
            psychological = None
        elif analysis_type == 'customer':
            result = sentiment_ai.analyze_customer_feedback(text_content)
            emotions = None
            psychological = None
        
        # Save analysis to database
        analysis = SentimentAnalysis(
            user_id=session.get('user_id'),
            text_content=text_content,
            analysis_type=analysis_type,
            sentiment=result.get('sentiment', 'neutral'),
            confidence_score=result.get('confidence_score', 0.5),
            emotions=json.dumps(emotions) if emotions else None,
            psychological_insights=json.dumps(psychological) if psychological else None,
            keywords=json.dumps(result.get('key_phrases', [])),
            category=result.get('feedback_category') or result.get('content_category') or 'general'
        )
        db.session.add(analysis)
        db.session.commit()
        
        return render_template('analysis_result.html', 
                             text=text_content,
                             result=result,
                             emotions=emotions,
                             psychological=psychological,
                             analysis_type=analysis_type,
                             analysis_id=analysis.id)
    
    return render_template('analyze.html')

@app.route('/batch_analyze', methods=['GET', 'POST'])
def batch_analyze():
    if request.method == 'POST':
        batch_name = request.form['batch_name']
        texts_input = request.form['texts_input']
        
        # Split texts by line
        texts = [text.strip() for text in texts_input.split('\n') if text.strip()]
        
        if not texts:
            flash('Please provide at least one text to analyze!')
            return redirect(url_for('batch_analyze'))
        
        # Perform batch analysis
        batch_result = sentiment_ai.analyze_batch_sentiment(texts)
        
        # Save individual analyses
        individual_results = []
        for text in texts[:20]:  # Limit to 20 texts
            basic_result = sentiment_ai.analyze_basic_sentiment(text)
            analysis = SentimentAnalysis(
                user_id=session.get('user_id'),
                text_content=text,
                analysis_type='batch',
                sentiment=basic_result.get('sentiment', 'neutral'),
                confidence_score=basic_result.get('confidence_score', 0.5),
                keywords=json.dumps(basic_result.get('key_phrases', [])),
                category='batch_analysis'
            )
            db.session.add(analysis)
            individual_results.append(basic_result)
        
        # Save batch analysis
        batch_analysis = BatchAnalysis(
            user_id=session.get('user_id'),
            batch_name=batch_name,
            total_texts=len(texts),
            overall_sentiment=batch_result.get('overall_sentiment', 'neutral'),
            summary_insights=json.dumps(batch_result)
        )
        db.session.add(batch_analysis)
        db.session.commit()
        
        return render_template('batch_analyze.html',
                             batch_name=batch_name,
                             batch_result=batch_result,
                             individual_results=individual_results,
                             texts=texts,
                             batch_id=batch_analysis.id)
    
    return render_template('batch_analyze.html')

@app.route('/history')
def analysis_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    page = request.args.get('page', 1, type=int)
    analyses = SentimentAnalysis.query.filter_by(user_id=session['user_id']).order_by(SentimentAnalysis.created_at.desc()).paginate(
        page=page, per_page=10, error_out=False)
    
    return render_template('history.html', analyses=analyses)

@app.route('/analysis/<int:analysis_id>')
def view_analysis(analysis_id):
    analysis = SentimentAnalysis.query.get_or_404(analysis_id)
    
    # Parse JSON fields
    emotions = json.loads(analysis.emotions) if analysis.emotions else None
    psychological = json.loads(analysis.psychological_insights) if analysis.psychological_insights else None
    keywords = json.loads(analysis.keywords) if analysis.keywords else []
    
    return render_template('view_analysis.html',
                         analysis=analysis,
                         emotions=emotions,
                         psychological=psychological,
                         keywords=keywords)

@app.route('/trends')
def sentiment_trends():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get analyses from last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    analyses = SentimentAnalysis.query.filter(
        SentimentAnalysis.user_id == session['user_id'],
        SentimentAnalysis.created_at >= thirty_days_ago
    ).order_by(SentimentAnalysis.created_at.asc()).all()
    
    return render_template('trends.html', analyses=analyses)

@app.route('/api/sentiment_chart')
def sentiment_chart():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    analyses = SentimentAnalysis.query.filter_by(user_id=session['user_id']).all()
    sentiment_counts = Counter([analysis.sentiment for analysis in analyses])
    
    # Create pie chart
    plt.figure(figsize=(8, 6))
    colors = ['#28a745', '#dc3545', '#6c757d']  # green, red, gray
    plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), colors=colors, autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    
    # Convert to base64
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'chart': plot_url})

@app.route('/api/emotion_chart/<int:analysis_id>')
def emotion_chart(analysis_id):
    analysis = SentimentAnalysis.query.get_or_404(analysis_id)
    
    if not analysis.emotions:
        return jsonify({'error': 'No emotion data'}), 404
    
    emotions = json.loads(analysis.emotions)
    emotion_scores = emotions.get('emotion_scores', {})
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    emotions_list = list(emotion_scores.keys())
    scores = list(emotion_scores.values())
    
    colors = plt.cm.Set3(range(len(emotions_list)))
    plt.bar(emotions_list, scores, color=colors)
    plt.title('Emotion Analysis')
    plt.ylabel('Intensity Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert to base64
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'chart': plot_url})

@app.route('/test_ai')
def test_ai():
    """Test endpoint to check if AI is working"""
    try:
        test_result = sentiment_ai.analyze_basic_sentiment("I am feeling great today! This is wonderful.")
        return f"<h2>AI Test Successful!</h2><p><strong>Sentiment:</strong> {test_result['sentiment']}</p><p><strong>Confidence:</strong> {test_result['confidence_score']}</p>"
    except Exception as e:
        return f"<h2>AI Test Failed!</h2><p>Error: {str(e)}</p>"

# Initialize database
def init_db():
    with app.app_context():
        db.create_all()
        print("Database initialized successfully!")

if __name__ == '__main__':
    # Check if API key is set
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("⚠️  WARNING: Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual Gemini API key!")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        print("   Then replace the GEMINI_API_KEY variable in app.py")
    else:
        print("✅ Gemini API key configured!")
        
        # Test the AI connection
        try:
            test_model = genai.GenerativeModel('gemini-1.5-flash')
            test_response = test_model.generate_content("Say 'Sentiment AI is working!'")
            print("🧠 Sentiment AI connection test successful!")
        except Exception as e:
            print(f"❌ Sentiment AI connection test failed: {e}")
            print("   Please check your API key and internet connection")
    
    init_db()
    print("🚀 Starting Sentiment Analysis Application...")
    print("🧠 Visit http://localhost:5000 to start analyzing sentiments")
    print("🧪 Visit http://localhost:5000/test_ai to test AI functionality")
    app.run(debug=True,port=5002)
"""
Flask Review Application for Moderators.

Provides a web UI for moderators to review comments flagged for manual review.
"""

import functools
import logging
import os
from typing import Optional

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from src.review.database import ReviewDatabase
from src.review.schemas import ReviewLabels, SubmitReviewRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authentication credentials from environment
AUTH_USERNAME = os.environ.get("REVIEW_AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("REVIEW_AUTH_PASSWORD", "changeme")

# Initialize database
db = ReviewDatabase()


def check_auth(username: str, password: str) -> bool:
    """Check if username/password combination is valid."""
    return username == AUTH_USERNAME and password == AUTH_PASSWORD


def authenticate():
    """Send a 401 response that enables basic auth."""
    return (
        jsonify({"error": "Authentication required"}),
        401,
        {"WWW-Authenticate": 'Basic realm="Review System"'},
    )


def requires_auth(f):
    """Decorator for routes that require authentication."""

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


def create_review_app() -> Flask:
    """Create the Flask review application."""
    app = Flask(__name__)

    # HTML template for the review UI
    REVIEW_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Toxic Comment Review</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f5f5f5;
                color: #333;
                line-height: 1.6;
            }
            .container { max-width: 900px; margin: 0 auto; padding: 20px; }
            header { 
                background: #2c3e50; 
                color: white; 
                padding: 20px; 
                margin-bottom: 20px;
                border-radius: 8px;
            }
            header h1 { font-size: 24px; margin-bottom: 5px; }
            .stats { 
                display: flex; 
                gap: 20px; 
                font-size: 14px; 
                opacity: 0.9; 
            }
            .review-card {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                overflow: hidden;
            }
            .review-header {
                background: #ecf0f1;
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #ddd;
            }
            .review-id { font-weight: 600; color: #2c3e50; }
            .review-meta { font-size: 13px; color: #7f8c8d; }
            .review-content { padding: 20px; }
            .comment-text {
                background: #f9f9f9;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin-bottom: 20px;
                font-size: 15px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .predictions {
                margin-bottom: 20px;
                padding: 15px;
                background: #fafafa;
                border-radius: 4px;
            }
            .predictions h4 { margin-bottom: 10px; color: #7f8c8d; font-size: 13px; }
            .prediction-bars { display: flex; flex-wrap: wrap; gap: 10px; }
            .prediction-bar {
                flex: 1 1 calc(33% - 10px);
                min-width: 150px;
            }
            .prediction-label { font-size: 12px; margin-bottom: 3px; }
            .prediction-value {
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
                overflow: hidden;
            }
            .prediction-fill {
                height: 100%;
                background: #3498db;
                transition: width 0.3s;
            }
            .prediction-fill.high { background: #e74c3c; }
            .prediction-fill.medium { background: #f39c12; }
            .labels-form h4 { margin-bottom: 15px; color: #2c3e50; }
            .checkbox-group {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
            }
            .checkbox-item {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 10px;
                background: #f9f9f9;
                border-radius: 4px;
                cursor: pointer;
                transition: background 0.2s;
            }
            .checkbox-item:hover { background: #ecf0f1; }
            .checkbox-item input { width: 18px; height: 18px; cursor: pointer; }
            .checkbox-item label { cursor: pointer; font-size: 14px; }
            .actions {
                display: flex;
                gap: 10px;
                margin-top: 20px;
                padding-top: 15px;
                border-top: 1px solid #eee;
            }
            .btn {
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: opacity 0.2s;
            }
            .btn:hover { opacity: 0.9; }
            .btn-primary { background: #3498db; color: white; }
            .btn-success { background: #27ae60; color: white; }
            .btn-secondary { background: #95a5a6; color: white; }
            .btn-danger { background: #e74c3c; color: white; }
            .empty-state {
                text-align: center;
                padding: 60px 20px;
                background: white;
                border-radius: 8px;
            }
            .empty-state h2 { color: #27ae60; margin-bottom: 10px; }
            .nav-links { margin-top: 20px; }
            .nav-links a { color: #3498db; margin-right: 15px; }
            @media (max-width: 600px) {
                .checkbox-group { grid-template-columns: repeat(2, 1fr); }
                .stats { flex-direction: column; gap: 5px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Toxic Comment Review</h1>
                <div class="stats">
                    <span>Pending: {{ pending_count }}</span>
                    <span>Reviewed: {{ reviewed_count }}</span>
                </div>
                <div class="nav-links">
                    <a href="{{ url_for('review_pending') }}">Pending Reviews</a>
                    <a href="{{ url_for('review_stats') }}">Statistics</a>
                </div>
            </header>

            {% if reviews %}
                {% for review in reviews %}
                <div class="review-card">
                    <div class="review-header">
                        <span class="review-id">Review #{{ review.id }}</span>
                        <span class="review-meta">
                            {{ review.source }} | {{ review.model_version or 'unknown' }}
                        </span>
                    </div>
                    <div class="review-content">
                        <div class="comment-text">{{ review.comment_text }}</div>
                        
                        {% if review.original_predictions %}
                        <div class="predictions">
                            <h4>MODEL PREDICTIONS</h4>
                            <div class="prediction-bars">
                                {% for label, value in review.original_predictions.items() %}
                                <div class="prediction-bar">
                                    <div class="prediction-label">
                                        {{ label }}: {{ "%.1f"|format(value * 100) }}%
                                    </div>
                                    <div class="prediction-value">
                                        <div class="prediction-fill {% if value > 0.7 %}high{% elif value > 0.4 %}medium{% endif %}" 
                                             style="width: {{ value * 100 }}%"></div>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                        {% endif %}

                        <form action="{{ url_for('submit_review') }}" method="POST">
                            <input type="hidden" name="review_id" value="{{ review.id }}">
                            <div class="labels-form">
                                <h4>YOUR LABELS</h4>
                                <div class="checkbox-group">
                                    {% for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] %}
                                    <div class="checkbox-item">
                                        <input type="checkbox" name="{{ label }}" id="{{ label }}_{{ review.id }}" value="1"
                                            {% if review.original_predictions and review.original_predictions.get(label, 0) > 0.5 %}checked{% endif %}>
                                        <label for="{{ label }}_{{ review.id }}">{{ label.replace('_', ' ').title() }}</label>
                                    </div>
                                    {% endfor %}
                                </div>
                            </div>
                            <div class="actions">
                                <button type="submit" class="btn btn-success">Submit Review</button>
                                <button type="submit" name="action" value="skip" class="btn btn-secondary">Skip</button>
                            </div>
                        </form>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div class="empty-state">
                    <h2>All caught up!</h2>
                    <p>No pending reviews at the moment.</p>
                </div>
            {% endif %}
        </div>
    </body>
    </html>
    """

    STATS_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Review Statistics</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f5f5f5;
                color: #333;
                line-height: 1.6;
            }
            .container { max-width: 900px; margin: 0 auto; padding: 20px; }
            header { 
                background: #2c3e50; 
                color: white; 
                padding: 20px; 
                margin-bottom: 20px;
                border-radius: 8px;
            }
            header h1 { font-size: 24px; margin-bottom: 5px; }
            .nav-links { margin-top: 15px; }
            .nav-links a { color: #ecf0f1; margin-right: 15px; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stat-value { font-size: 36px; font-weight: 700; color: #2c3e50; }
            .stat-label { color: #7f8c8d; font-size: 14px; }
            .section {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .section h3 { margin-bottom: 15px; color: #2c3e50; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f9f9f9; font-weight: 600; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Review Statistics</h1>
                <div class="nav-links">
                    <a href="{{ url_for('review_pending') }}">Back to Reviews</a>
                </div>
            </header>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ stats.total_pending }}</div>
                    <div class="stat-label">Pending</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ stats.total_reviewed }}</div>
                    <div class="stat-label">Reviewed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ stats.status_counts.get('skipped', 0) }}</div>
                    <div class="stat-label">Skipped</div>
                </div>
            </div>

            <div class="section">
                <h3>Daily Reviews (Last 7 Days)</h3>
                <table>
                    <thead>
                        <tr><th>Date</th><th>Reviews</th></tr>
                    </thead>
                    <tbody>
                        {% for day in stats.daily_reviews %}
                        <tr>
                            <td>{{ day.date }}</td>
                            <td>{{ day.count }}</td>
                        </tr>
                        {% else %}
                        <tr><td colspan="2">No reviews in the last 7 days</td></tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <div class="section">
                <h3>Top Moderators</h3>
                <table>
                    <thead>
                        <tr><th>Moderator</th><th>Reviews</th></tr>
                    </thead>
                    <tbody>
                        {% for mod in stats.top_moderators %}
                        <tr>
                            <td>{{ mod.moderator }}</td>
                            <td>{{ mod.count }}</td>
                        </tr>
                        {% else %}
                        <tr><td colspan="2">No moderators yet</td></tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    @app.route("/")
    @requires_auth
    def index():
        return redirect(url_for("review_pending"))

    @app.route("/review/pending")
    @requires_auth
    def review_pending():
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 5))
        offset = (page - 1) * limit

        reviews = db.get_pending_reviews(limit=limit, offset=offset)
        pending_count = db.get_pending_count()
        stats = db.get_statistics()

        return render_template_string(
            REVIEW_TEMPLATE,
            reviews=reviews,
            pending_count=pending_count,
            reviewed_count=stats.get("total_reviewed", 0),
            page=page,
            limit=limit,
        )

    @app.route("/review/submit", methods=["POST"])
    @requires_auth
    def submit_review():
        review_id = int(request.form.get("review_id"))
        action = request.form.get("action", "submit")
        moderator_id = request.authorization.username

        if action == "skip":
            db.skip_review(review_id, moderator_id)
        else:
            labels = ReviewLabels(
                toxic=1 if request.form.get("toxic") else 0,
                severe_toxic=1 if request.form.get("severe_toxic") else 0,
                obscene=1 if request.form.get("obscene") else 0,
                threat=1 if request.form.get("threat") else 0,
                insult=1 if request.form.get("insult") else 0,
                identity_hate=1 if request.form.get("identity_hate") else 0,
            )
            db.submit_review(review_id, labels.model_dump(), moderator_id)

        return redirect(url_for("review_pending"))

    @app.route("/review/stats")
    @requires_auth
    def review_stats():
        stats = db.get_statistics()
        return render_template_string(STATS_TEMPLATE, stats=stats)

    # API endpoints for programmatic access
    @app.route("/api/reviews/pending", methods=["GET"])
    @requires_auth
    def api_pending_reviews():
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 10))
        offset = (page - 1) * limit

        reviews = db.get_pending_reviews(limit=limit, offset=offset)
        pending_count = db.get_pending_count()

        return jsonify(
            {
                "reviews": reviews,
                "total_pending": pending_count,
                "page": page,
                "limit": limit,
            }
        )

    @app.route("/api/reviews/submit", methods=["POST"])
    @requires_auth
    def api_submit_review():
        data = request.get_json()
        review_id = data.get("review_id")
        labels = data.get("labels", {})
        moderator_id = request.authorization.username

        labels_obj = ReviewLabels(**labels)
        success = db.submit_review(review_id, labels_obj.model_dump(), moderator_id)

        return jsonify(
            {
                "success": success,
                "message": "Review submitted" if success else "Review not found",
                "review_id": review_id,
            }
        )

    @app.route("/api/reviews/stats", methods=["GET"])
    @requires_auth
    def api_stats():
        stats = db.get_statistics()
        return jsonify(stats)

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy"})

    return app


# Create the app instance
app = create_review_app()


if __name__ == "__main__":
    # Initialize the database schema
    db.initialize_schema()

    # Run the development server
    app.run(
        host=os.environ.get("REVIEW_HOST", "0.0.0.0"),
        port=int(os.environ.get("REVIEW_PORT", 5001)),
        debug=True,
    )

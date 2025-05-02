import os
import re
import pandas as pd
import emoji
import nltk
nltk.data.path = ["C:/Users/achin/Documents/DSA/NLP"]
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
import threading
import customtkinter as ctk
from tkinter import filedialog, StringVar, scrolledtext
import tkinter as tk

# Removed: from emosent import get_emoji_sentiment_rank_multiple

# Simple emoji sentiment map
emoji_sentiment_map = {
    '😀': 1, '😄': 1, '😊': 1, '😍': 1,
    '😢': -1, '😭': -1, '😠': -1, '😡': -1,
    '😐': 0, '😶': 0
}

def get_emoji_sentiment_score(text):
    if not isinstance(text, str):
        return 0
    score = 0
    count = 0
    for char in text:
        if char in emoji_sentiment_map:
            score += emoji_sentiment_map[char]
            count += 1
    return score / count if count > 0 else 0

# nltk_data_dir = os.path.join("C:", "Users", "achin","Documents","DSA","NLP", "nltk_data")
nltk_data_dir = "./nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)  # Create directory if missing
nltk.data.path = [nltk_data_dir]  # Tell NLTK to use this directory
# Download necessary NLTK resources (uncomment first time)
nltk.download('punkt', download_dir=nltk_data_dir, force=True)
nltk.download('punkt_tab', download_dir=nltk_data_dir, force=True)
nltk.download('vader_lexicon', download_dir=nltk_data_dir, force=True)
nltk.download('stopwords', download_dir=nltk_data_dir, force=True)
print("Punkt is installed correctly!")

try:
    nltk.data.find('tokenizers/punkt')
    print("SUCCESS: Punkt tokenizer installed!")
except LookupError:
    print("ERROR: Punkt still missing. Check permissions or path.")

    
# Set CustomTkinter appearance
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class MatplotlibCanvas(FigureCanvasTkAgg):
    def __init__(self, figure, master=None):
        super().__init__(figure, master)
        self.draw()


class WhatsAppAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("WhatsApp Chat Analyzer")
        self.geometry("1200x800")

        # Create variables
        self.file_path_var = StringVar()
        self.chat_data = None
        self.analysis_results = None

        # Create main layout
        self.create_widgets()

    def create_widgets(self):
        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # File selection area
        self.file_frame = ctk.CTkFrame(self.main_frame)
        self.file_frame.pack(fill="x", padx=10, pady=10)

        self.file_label = ctk.CTkLabel(self.file_frame, text="Chat File:")
        self.file_label.pack(side="left", padx=5)

        self.file_entry = ctk.CTkEntry(self.file_frame, textvariable=self.file_path_var, width=400)
        self.file_entry.pack(side="left", padx=5, fill="x", expand=True)

        self.browse_button = ctk.CTkButton(self.file_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(side="left", padx=5)

        # Analysis button
        self.analyze_button = ctk.CTkButton(self.main_frame, text="Analyze Chat",
                                            command=self.analyze_chat, height=40)
        self.analyze_button.pack(fill="x", padx=10, pady=10)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        self.progress_bar.pack_forget()  # Hide initially

        # Status label
        self.status_label = ctk.CTkLabel(self.main_frame, text="")
        self.status_label.pack(padx=10, pady=5)

        # Tabview for results
        self.results_tabview = ctk.CTkTabview(self.main_frame)
        self.results_tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Create tabs
        self.stats_tab = self.results_tabview.add("Statistics")
        self.user_activity_tab = self.results_tabview.add("User Activity")
        self.time_analysis_tab = self.results_tabview.add("Time Analysis")
        self.sentiment_tab = self.results_tabview.add("Sentiment")
        self.word_cloud_tab = self.results_tabview.add("Word Cloud")
        self.topic_tab = self.results_tabview.add("Topics")
        self.emoji_sentiment_tab = self.results_tabview.add("Emoji Sentiment")  # New tab for emoji sentiment

        # Create scrollable frames for each tab
        self.stats_frame = ctk.CTkScrollableFrame(self.stats_tab)
        self.stats_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.user_activity_frame = ctk.CTkScrollableFrame(self.user_activity_tab)
        self.user_activity_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.time_analysis_frame = ctk.CTkScrollableFrame(self.time_analysis_tab)
        self.time_analysis_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.sentiment_frame = ctk.CTkScrollableFrame(self.sentiment_tab)
        self.sentiment_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.word_cloud_frame = ctk.CTkScrollableFrame(self.word_cloud_tab)
        self.word_cloud_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.topic_frame = ctk.CTkScrollableFrame(self.topic_tab)
        self.topic_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.emoji_sentiment_frame = ctk.CTkScrollableFrame(self.emoji_sentiment_tab)
        self.emoji_sentiment_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Add a dummy label to the emoji sentiment tab initially
        self.emoji_placeholder = ctk.CTkLabel(self.emoji_sentiment_frame,
                                              text="Emoji sentiment analysis will appear here after processing.")
        self.emoji_placeholder.pack(padx=10, pady=10)

        # Add a generate dummy data button
        self.dummy_data_frame = ctk.CTkFrame(self.main_frame)
        self.dummy_data_frame.pack(fill="x", padx=10, pady=5)

        self.dummy_data_button = ctk.CTkButton(
            self.dummy_data_frame,
            text="Generate Test Data",
            command=self.generate_test_data,
            fg_color="#2E7D32",  # Green color
            hover_color="#1B5E20"
        )
        self.dummy_data_button.pack(side="left", padx=5, pady=5)

        self.dummy_data_label = ctk.CTkLabel(self.dummy_data_frame, text="")
        self.dummy_data_label.pack(side="left", padx=5, pady=5)

        # Add theme switcher
        self.appearance_frame = ctk.CTkFrame(self.main_frame)
        self.appearance_frame.pack(fill="x", padx=10, pady=5)

        self.appearance_label = ctk.CTkLabel(self.appearance_frame, text="Theme:")
        self.appearance_label.pack(side="left", padx=5, pady=5)

        self.appearance_option = ctk.CTkOptionMenu(
            self.appearance_frame,
            values=["System", "Light", "Dark"],
            command=self.change_appearance_mode
        )
        self.appearance_option.pack(side="left", padx=5, pady=5)

        # Set default appearance mode
        self.appearance_option.set("System")

    def generate_test_data(self):
        """Generate dummy WhatsApp chat data for testing"""
        try:
            filename = "dummy_whatsapp_chat.txt"
            generate_dummy_data(filename)
            self.file_path_var.set(os.path.abspath(filename))
            self.dummy_data_label.configure(text=f"Test data generated: {filename}")
        except Exception as e:
            self.dummy_data_label.configure(text=f"Error generating test data: {str(e)}")

    def change_appearance_mode(self, new_appearance_mode):
        """Change the app's appearance mode"""
        ctk.set_appearance_mode(new_appearance_mode)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Open WhatsApp Chat",
            filetypes=[("Text Files", "*.txt")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def analyze_sentiment(df):
        sia = SentimentIntensityAnalyzer()

        # Function to extract emoji sentiment
        def get_emoji_sentiment(text):
            if not isinstance(text, str):
                return 0

            emoji_data = get_emoji_sentiment_score(text)
            if not emoji_data:
                return 0

            # Average the sentiment scores of all emojis in the text
            total_score = sum(item['emoji_sentiment_rank'].get('sentiment_score', 0) for item in emoji_data)
            count = len(emoji_data)
            return total_score / count if count > 0 else 0

        # Apply text sentiment analysis
        df['text_sentiment_score'] = df['message'].apply(
            lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0
        )

        # Apply emoji sentiment analysis
        df['emoji_sentiment_score'] = df['message'].apply(get_emoji_sentiment)

        # Combine text and emoji sentiment (with weights if desired)
        df['sentiment_score'] = df.apply(
            lambda row: (row['text_sentiment_score'] * 0.7) + (row['emoji_sentiment_score'] * 0.3)
            if row['emoji_sentiment_score'] > 0 else row['text_sentiment_score'],
            axis=1
        )

        # Categorize sentiment
        df['sentiment'] = df['sentiment_score'].apply(
            lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
        )

        # Aggregate results
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']

        # Sentiment by user
        user_sentiment = df.groupby(['author', 'sentiment']).size().reset_index(name='count')

        return {
            'overall': sentiment_counts,
            'by_user': user_sentiment
        }
    def analyze_chat(self):
        file_path = self.file_path_var.get()
        if not file_path:
            self.status_label.configure(text="Please select a chat file first.")
            return

        # Show loading indicator
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        self.status_label.configure(text="Processing chat data...")

        # Clear previous results
        self.clear_results()

        # Start analysis in a separate thread
        self.analysis_thread = threading.Thread(target=self.run_analysis, args=(file_path,))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def run_analysis(self, file_path):
        try:
            # Process chat data
            self.chat_data = preprocess(file_path)

            # Update progress
            self.after(100, lambda: self.progress_bar.set(0.3))
            self.after(100, lambda: self.status_label.configure(text="Analyzing data..."))

            # Perform analysis
            self.analysis_results = perform_analysis(self.chat_data)

            # Update progress
            self.after(100, lambda: self.progress_bar.set(0.7))
            self.after(100, lambda: self.status_label.configure(text="Generating visualizations..."))

            # Display results in the main thread
            self.after(100, self.display_results)

        except Exception as e:
            # Capture the error message
            error_message = str(e)
            # Use a function instead of lambda to avoid the variable scope issue
            self.after(100, lambda: self.show_error(error_message))
            self.after(100, lambda: self.progress_bar.pack_forget())

    def show_error(self, message):
        """Helper method to display error messages"""
        self.status_label.configure(text=f"Error: {message}")

    def clear_results(self):
        # Clear all frames
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        for widget in self.user_activity_frame.winfo_children():
            widget.destroy()
        for widget in self.time_analysis_frame.winfo_children():
            widget.destroy()
        for widget in self.sentiment_frame.winfo_children():
            widget.destroy()
        for widget in self.word_cloud_frame.winfo_children():
            widget.destroy()
        for widget in self.topic_frame.winfo_children():
            widget.destroy()

    def display_results(self):
        # Hide loading indicator
        self.progress_bar.pack_forget()

        # Display statistics
        self.display_statistics(self.analysis_results['stats'])

        # Display visualizations
        self.display_user_activity(self.analysis_results['user_activity'])
        self.display_time_analysis(self.analysis_results['time_analysis'])
        self.display_sentiment_analysis(self.analysis_results['sentiment'])
        self.display_word_cloud(self.analysis_results['word_frequencies'])
        self.display_topic_modeling(self.analysis_results['topics'])

        self.status_label.configure(text="Analysis completed successfully!")

    def display_statistics(self, stats):
        # Create a grid for statistics
        row = 0
        for key, value in stats.items():
            # Format the key
            formatted_key = key.replace('_', ' ').title()

            # Format the value
            if key == 'date_range':
                formatted_value = f"{value[0]} to {value[1]}"
            else:
                formatted_value = str(value)

            # Create labels
            key_label = ctk.CTkLabel(self.stats_frame, text=f"{formatted_key}:",
                                     font=ctk.CTkFont(weight="bold"))
            key_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)

            value_label = ctk.CTkLabel(self.stats_frame, text=formatted_value)
            value_label.grid(row=row, column=1, sticky="w", padx=5, pady=2)

            row += 1

    def display_user_activity(self, user_activity):
        # Create canvas for user message count chart
        fig1 = Figure(figsize=(10, 6), dpi=100)

        # Sort by message count
        user_activity_sorted = user_activity.sort_values('message_count', ascending=False)

        # Plot
        ax1 = fig1.add_subplot(111)
        bars = ax1.bar(user_activity_sorted['author'], user_activity_sorted['message_count'])

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom')

        ax1.set_title('Messages per User')
        ax1.set_xlabel('User')
        ax1.set_ylabel('Number of Messages')
        ax1.tick_params(axis='x', rotation=45)
        fig1.tight_layout()

        # Create canvas widget
        canvas1 = MatplotlibCanvas(fig1, master=self.user_activity_frame)
        canvas_widget1 = canvas1.get_tk_widget()
        canvas_widget1.pack(fill="both", expand=True, padx=5, pady=5)

        # Create canvas for words per user chart
        fig2 = Figure(figsize=(10, 6), dpi=100)

        # Sort by word count
        user_activity_sorted = user_activity.sort_values('word_count', ascending=False)

        # Plot
        ax2 = fig2.add_subplot(111)
        bars = ax2.bar(user_activity_sorted['author'], user_activity_sorted['word_count'])

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom')

        ax2.set_title('Words per User')
        ax2.set_xlabel('User')
        ax2.set_ylabel('Number of Words')
        ax2.tick_params(axis='x', rotation=45)
        fig2.tight_layout()

        # Create canvas widget
        canvas2 = MatplotlibCanvas(fig2, master=self.user_activity_frame)
        canvas_widget2 = canvas2.get_tk_widget()
        canvas_widget2.pack(fill="both", expand=True, padx=5, pady=5)

    def display_time_analysis(self, time_analysis):
        # Create canvas for hourly activity
        fig1 = Figure(figsize=(10, 5), dpi=100)

        # Plot
        ax1 = fig1.add_subplot(111)
        ax1.bar(time_analysis['hourly']['hour'], time_analysis['hourly']['message_count'])
        ax1.set_title('Messages by Hour of Day')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Number of Messages')
        ax1.set_xticks(range(0, 24))
        fig1.tight_layout()

        # Create canvas widget
        canvas1 = MatplotlibCanvas(fig1, master=self.time_analysis_frame)
        canvas_widget1 = canvas1.get_tk_widget()
        canvas_widget1.pack(fill="both", expand=True, padx=5, pady=5)

        # Create canvas for daily activity
        fig2 = Figure(figsize=(10, 5), dpi=100)

        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_df = time_analysis['daily'].copy()
        daily_df['day_name'] = pd.Categorical(daily_df['day_name'], categories=days_order, ordered=True)
        daily_df = daily_df.sort_values('day_name')

        # Plot
        ax2 = fig2.add_subplot(111)
        ax2.bar(daily_df['day_name'], daily_df['message_count'])
        ax2.set_title('Messages by Day of Week')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Number of Messages')
        fig2.tight_layout()

        # Create canvas widget
        canvas2 = MatplotlibCanvas(fig2, master=self.time_analysis_frame)
        canvas_widget2 = canvas2.get_tk_widget()
        canvas_widget2.pack(fill="both", expand=True, padx=5, pady=5)

        # Create canvas for date activity (time series)
        fig3 = Figure(figsize=(10, 5), dpi=100)

        # Plot
        ax3 = fig3.add_subplot(111)
        ax3.plot(time_analysis['date']['date'], time_analysis['date']['message_count'])
        ax3.set_title('Messages over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Messages')
        ax3.tick_params(axis='x', rotation=45)
        fig3.tight_layout()

        # Create canvas widget
        canvas3 = MatplotlibCanvas(fig3, master=self.time_analysis_frame)
        canvas_widget3 = canvas3.get_tk_widget()
        canvas_widget3.pack(fill="both", expand=True, padx=5, pady=5)

    def display_sentiment_analysis(self, sentiment):
        # Create canvas for overall sentiment
        fig1 = Figure(figsize=(8, 6), dpi=100)

        # Plot
        ax1 = fig1.add_subplot(111)
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}

        # Sort by sentiment (negative, neutral, positive)
        sentiment_order = ['negative', 'neutral', 'positive']
        overall_df = sentiment['overall'].copy()
        overall_df['sentiment'] = pd.Categorical(overall_df['sentiment'], categories=sentiment_order, ordered=True)
        overall_df = overall_df.sort_values('sentiment')

        bars = ax1.bar(overall_df['sentiment'], overall_df['count'],
                       color=[colors[s] for s in overall_df['sentiment']])

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom')

        ax1.set_title('Overall Sentiment Distribution')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Number of Messages')
        fig1.tight_layout()

        # Create canvas widget
        canvas1 = MatplotlibCanvas(fig1, master=self.sentiment_frame)
        canvas_widget1 = canvas1.get_tk_widget()
        canvas_widget1.pack(fill="both", expand=True, padx=5, pady=5)

        # Create canvas for sentiment by user
        fig2 = Figure(figsize=(10, 6), dpi=100)

        # Plot
        ax2 = fig2.add_subplot(111)

        # Pivot the data
        user_sentiment_pivot = sentiment['by_user'].pivot(index='author', columns='sentiment', values='count').fillna(0)

        # Ensure all sentiments are present
        for s in sentiment_order:
            if s not in user_sentiment_pivot.columns:
                user_sentiment_pivot[s] = 0

        # Sort by total message count
        user_sentiment_pivot['total'] = user_sentiment_pivot.sum(axis=1)
        user_sentiment_pivot = user_sentiment_pivot.sort_values('total', ascending=False)
        user_sentiment_pivot = user_sentiment_pivot.drop('total', axis=1)

        # Select only sentiment columns in the right order
        user_sentiment_pivot = user_sentiment_pivot[sentiment_order]

        # Plot stacked bar
        user_sentiment_pivot.plot(kind='bar', stacked=True, ax=ax2, color=[colors[s] for s in sentiment_order])

        ax2.set_title('Sentiment by User')
        ax2.set_xlabel('User')
        ax2.set_ylabel('Number of Messages')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Sentiment')
        fig2.tight_layout()

        # Create canvas widget
        canvas2 = MatplotlibCanvas(fig2, master=self.sentiment_frame)
        canvas_widget2 = canvas2.get_tk_widget()
        canvas_widget2.pack(fill="both", expand=True, padx=5, pady=5)

    def display_word_cloud(self, word_frequencies):
        # Create canvas for word cloud
        if word_frequencies:
            fig1 = Figure(figsize=(10, 8), dpi=100)

            # Generate word cloud
            wordcloud = WordCloud(width=800, height=600, background_color='white',
                                  max_words=100, contour_width=3, contour_color='steelblue')
            wordcloud.generate_from_frequencies(word_frequencies)

            # Display word cloud
            ax1 = fig1.add_subplot(111)
            ax1.imshow(wordcloud, interpolation='bilinear')
            ax1.axis('off')
            fig1.tight_layout()

            # Create canvas widget
            canvas1 = MatplotlibCanvas(fig1, master=self.word_cloud_frame)
            canvas_widget1 = canvas1.get_tk_widget()
            canvas_widget1.pack(fill="both", expand=True, padx=5, pady=5)

            # Also display top words as a bar chart
            fig2 = Figure(figsize=(10, 6), dpi=100)

            # Get top 20 words
            top_words = dict(sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:20])

            # Plot
            ax2 = fig2.add_subplot(111)
            bars = ax2.barh(list(reversed(list(top_words.keys()))), list(reversed(list(top_words.values()))))

            # Add data labels
            for bar in bars:
                width = bar.get_width()
                ax2.text(width + 0.5, bar.get_y() + bar.get_height() / 2.,
                         f'{int(width)}', ha='left', va='center')

            ax2.set_title('Top 20 Words')
            ax2.set_xlabel('Frequency')
            fig2.tight_layout()

            # Create canvas widget
            canvas2 = MatplotlibCanvas(fig2, master=self.word_cloud_frame)
            canvas_widget2 = canvas2.get_tk_widget()
            canvas_widget2.pack(fill="both", expand=True, padx=5, pady=5)
        else:
            # Display message if no word frequencies
            label = ctk.CTkLabel(self.word_cloud_frame, text="No word frequency data available.")
            label.pack(padx=10, pady=10)

    def display_topic_modeling(self, topics):
        # Create a grid for topics
        row = 0
        for topic, words in topics.items():
            # Format the topic
            formatted_topic = topic.replace('_', ' ').title()

            # Create labels
            topic_label = ctk.CTkLabel(self.topic_frame, text=f"{formatted_topic}:",
                                       font=ctk.CTkFont(weight="bold"))
            topic_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)

            words_label = ctk.CTkLabel(self.topic_frame, text=", ".join(words), wraplength=500)
            words_label.grid(row=row, column=1, sticky="w", padx=5, pady=2)

            row += 1


# Preprocessing functions
def preprocess(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Define regex pattern to extract date, time, author and message
    # This pattern works for the format: "MM/DD/YY, HH:MM AM/PM - Author: Message"
    pattern = r'(\d+/\d+/\d+,\s\d+:\d+\s[APMapm]+)\s-\s([^:]+):\s(.*)'

    # Extract data
    messages = re.findall(pattern, data)

    # Create DataFrame
    df = pd.DataFrame(messages, columns=['date_time', 'author', 'message'])

    # Convert date_time to datetime
    df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)

    # Extract additional features
    df['date'] = df['date_time'].dt.date
    df['time'] = df['date_time'].dt.time
    df['hour'] = df['date_time'].dt.hour
    df['day_name'] = df['date_time'].dt.day_name()

    # Detect media messages
    df['is_media'] = df['message'].apply(lambda x: 1 if '<Media omitted>' in x else 0)

    # Extract URLs
    url_pattern = r'(https?://\S+)'
    df['has_url'] = df['message'].apply(lambda x: 1 if re.search(url_pattern, x) else 0)

    # Extract emojis
    df['emojis'] = df['message'].apply(extract_emojis)

    return df


def extract_emojis(text):
    if not isinstance(text, str):
        return ""
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)


# Analysis functions
def perform_analysis(df):
    results = {}

    # Basic statistics
    results['stats'] = get_basic_stats(df)

    # User activity analysis
    results['user_activity'] = analyze_user_activity(df)

    # Time analysis
    results['time_analysis'] = analyze_time_patterns(df)

    # Sentiment analysis
    results['sentiment'] = analyze_sentiment(df)

    # Word frequency analysis
    results['word_frequencies'] = analyze_word_frequencies(df)

    # Topic modeling
    results['topics'] = perform_topic_modeling(df)

    return results


def get_basic_stats(df):
    stats = {
        'total_messages': len(df),
        'media_messages': df['is_media'].sum(),
        'url_messages': df['has_url'].sum(),
        'total_users': df['author'].nunique(),
        'date_range': (df['date_time'].min().date(), df['date_time'].max().date()),
        'average_messages_per_day': round(len(df) / (df['date_time'].max().date() - df['date_time'].min().date()).days,
                                          2) if (df['date_time'].max().date() - df[
            'date_time'].min().date()).days > 0 else len(df)
    }
    return stats


def analyze_user_activity(df):
    # Messages per user
    user_messages = df['author'].value_counts().reset_index()
    user_messages.columns = ['author', 'message_count']

    # Words per user
    df['word_count'] = df['message'].apply(lambda x: len(word_tokenize(x)) if isinstance(x, str) else 0)
    words_per_user = df.groupby('author')['word_count'].sum().reset_index()

    # Merge the data
    user_activity = pd.merge(user_messages, words_per_user, on='author')

    return user_activity


def analyze_time_patterns(df):
    # Messages by hour
    hourly_activity = df.groupby('hour').size().reset_index(name='message_count')

    # Messages by day
    daily_activity = df.groupby('day_name').size().reset_index(name='message_count')

    # Messages by date
    date_activity = df.groupby('date').size().reset_index(name='message_count')

    return {
        'hourly': hourly_activity,
        'daily': daily_activity,
        'date': date_activity
    }


def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment analysis
    df['sentiment_score'] = df['message'].apply(
        lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )

    # Categorize sentiment
    df['sentiment'] = df['sentiment_score'].apply(
        lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
    )

    # Aggregate results
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']

    # Sentiment by user
    user_sentiment = df.groupby(['author', 'sentiment']).size().reset_index(name='count')

    return {
        'overall': sentiment_counts,
        'by_user': user_sentiment
    }


def analyze_word_frequencies(df):
    # Combine all messages
    all_messages = ' '.join(df['message'].dropna())

    # Tokenize
    tokens = word_tokenize(all_messages.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Count frequencies
    word_freq = Counter(filtered_tokens)

    return word_freq


def perform_topic_modeling(df):
    # Simplified topic modeling - in a real app, you might use LDA or BERTopic
    # This is a placeholder for the actual implementation

    # Get most common words for each user
    user_topics = {}

    for author in df['author'].unique():
        author_messages = ' '.join(df[df['author'] == author]['message'].dropna())
        tokens = word_tokenize(author_messages.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 3]
        word_freq = Counter(filtered_tokens)
        top_words = [word for word, _ in word_freq.most_common(5)]
        user_topics[f"Topic for {author}"] = top_words

    # Add some general topics
    all_messages = ' '.join(df['message'].dropna())
    tokens = word_tokenize(all_messages.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 3]
    word_freq = Counter(filtered_tokens)

    # Get top words for different time periods
    if len(df) > 100:
        df_sorted = df.sort_values('date_time')
        segments = 3
        segment_size = len(df) // segments

        for i in range(segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < segments - 1 else len(df)
            segment_df = df_sorted.iloc[start_idx:end_idx]

            start_date = segment_df['date_time'].min().date()
            end_date = segment_df['date_time'].max().date()

            segment_messages = ' '.join(segment_df['message'].dropna())
            tokens = word_tokenize(segment_messages.lower())
            filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 3]
            word_freq = Counter(filtered_tokens)
            top_words = [word for word, _ in word_freq.most_common(5)]

            user_topics[f"Topic from {start_date} to {end_date}"] = top_words

    return user_topics


# Generate dummy WhatsApp chat data for testing
def generate_dummy_data(filename="dummy_whatsapp_chat.txt", num_messages=200):
    users = ["Alice", "Bob", "Charlie", "David", "Eva"]
    message_templates = [
        "Hey, how are you?",
        "Let's meet at 5 PM.",
        "Did you see the game last night?",
        "I will send the report by tomorrow.",
        "Can you help me with the project?",
        "Sure, what do you need?",
        "Yes, it was amazing! 😄",
        "No, I couldn't make it.",
        "Check out this link: https://example.com",
        "<Media omitted>"
    ]

    # Generate random dates within the last 10 days
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=10)

    # Generate random timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, periods=num_messages)
    timestamps = timestamps.sort_values()

    # Format the chat data
    chat_lines = []
    for ts in timestamps:
        user = np.random.choice(users)
        message = np.random.choice(message_templates)
        formatted_ts = ts.strftime("%m/%d/%y, %I:%M %p")
        chat_line = f"{formatted_ts} - {user}: {message}"
        chat_lines.append(chat_line)

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(chat_lines))

    return filename


if __name__ == "__main__":
    # Uncomment to generate dummy data
    # generate_dummy_data()

    app = WhatsAppAnalyzerApp()
    app.mainloop()

from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import json
import traceback
import base64
from io import BytesIO
import PyPDF2
import docx
import pandas as pd
from datetime import datetime, date
import requests
import re
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import time
import os
import tiktoken

client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

app = Flask(__name__)
CORS(app)

chat_histories = {}
conversation_metadata = {}

# Token tracking storage
daily_token_usage = {}

# Token limits for context window management - ONLY for context, not response
MAX_CONTEXT_TOKENS = 3000
TOKENS_PER_CHAR = 4

def get_token_counter():
    """Get or create token counter for today"""
    today = str(date.today())
    if today not in daily_token_usage:
        daily_token_usage[today] = {
            "date": today,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
            "cost_estimate": 0.0
        }
    return daily_token_usage[today]

def count_tokens_tiktoken(text, model="gpt-4o-mini"):
    """Count tokens using tiktoken for accurate counting"""
    try:
        # Get the encoding for the model
        if model.startswith("gpt-4"):
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif model.startswith("gpt-3.5"):
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        if isinstance(text, list):
            # Handle message array format
            total_tokens = 0
            for item in text:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        total_tokens += len(encoding.encode(item.get("text", "")))
                    # Skip image tokens for now (they're more complex to calculate)
                else:
                    total_tokens += len(encoding.encode(str(item)))
            return total_tokens
        else:
            return len(encoding.encode(str(text)))
    except Exception as e:
        print(f"Token counting error: {e}")
        # Fallback to character-based estimation
        return estimate_tokens(text)

def estimate_cost(input_tokens, output_tokens, model):
    """Estimate cost based on token usage and model pricing"""
    # Pricing per 1M tokens (as of 2024)
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
    }
    
    model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
    
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    
    return input_cost + output_cost

def update_token_usage(input_tokens, output_tokens, model):
    """Update daily token usage tracking"""
    counter = get_token_counter()
    
    counter["input_tokens"] += input_tokens
    counter["output_tokens"] += output_tokens
    counter["total_tokens"] += (input_tokens + output_tokens)
    counter["requests"] += 1
    
    # Update cost estimate
    request_cost = estimate_cost(input_tokens, output_tokens, model)
    counter["cost_estimate"] += request_cost
    
    # Clean up old data (keep only last 30 days)
    today = date.today()
    old_dates = [d for d in daily_token_usage.keys() 
                 if (today - datetime.strptime(d, "%Y-%m-%d").date()).days > 30]
    for old_date in old_dates:
        del daily_token_usage[old_date]
    
    print(f"📊 Token usage updated: +{input_tokens} input, +{output_tokens} output (${request_cost:.4f})")
    return counter

# System prompts for different modes
SYSTEM_PROMPTS = {
    "general": None,
    "coding": """You are an expert software engineer and coding mentor. When writing code:
- Write clean, readable, well-commented code
- Follow best practices and modern conventions
- Explain your reasoning step-by-step
- Include error handling and edge cases
- Suggest optimizations and improvements
- Use TypeScript when applicable for better type safety
- Provide working, tested examples
- Explain potential issues or limitations
Always structure your response with: 1) Explanation, 2) Code, 3) Usage example when relevant.""",
    
    "creative": """You are a creative writing assistant and brainstorming partner. Help with:
- Creative writing (stories, poems, scripts)
- Brainstorming ideas and concepts
- Character development and worldbuilding
- Writing style and narrative techniques
- Creative problem-solving approaches
Be imaginative, inspiring, and offer multiple creative perspectives. Focus on originality and artistic expression.""",
    
    "analysis": """You are a data analyst and research expert. When analyzing:
- Break down complex problems systematically
- Provide structured, logical analysis
- Use data-driven reasoning when possible
- Identify patterns, trends, and insights
- Consider multiple perspectives and potential biases
- Present findings clearly with supporting evidence
- Suggest actionable recommendations
Structure your analysis with clear sections and bullet points for key findings.""",
    
    "learning": """You are a patient, knowledgeable tutor and educational mentor. When teaching:
- Explain concepts clearly from first principles
- Use analogies and examples to illustrate points
- Break complex topics into digestible steps
- Encourage questions and active learning
- Provide practice exercises when appropriate
- Adapt explanations to the user's level
- Connect new information to existing knowledge
Always check understanding and offer to elaborate on any unclear points."""
}

def get_system_prompt(mode, has_search_results=False):
    """Get the system prompt for the specified mode"""
    base_prompt = SYSTEM_PROMPTS.get(mode)
    
    if has_search_results:
        search_addon = "\n\nYou have access to current web search results. Use them to provide up-to-date, accurate information. Always cite your sources when using search results."
        return (base_prompt or "") + search_addon
    
    return base_prompt

def get_temperature_for_mode(mode):
    """Get optimal temperature setting for each mode"""
    temperatures = {
        "general": 0.3,
        "coding": 0.1,
        "creative": 0.8,
        "analysis": 0.2,
        "learning": 0.3
    }
    return temperatures.get(mode, 0.3)

def should_search(message, mode):
    """Determine if a message should trigger web search"""
    # Search triggers
    search_indicators = [
        "search", "find", "look up", "what's new", "latest", "recent", "current", 
        "today", "2024", "2025", "now", "update", "news", "price", "cost",
        "weather", "stock", "trending", "happening", "breaking"
    ]
    
    # Don't search for these types of queries
    no_search_indicators = [
        "code", "write", "create", "make", "build", "develop", "program",
        "explain", "how to", "what is", "define", "meaning", "concept"
    ]
    
    message_lower = message.lower()
    
    # Check if message contains search indicators
    has_search_trigger = any(indicator in message_lower for indicator in search_indicators)
    
    # Check if message should NOT be searched
    has_no_search = any(indicator in message_lower for indicator in no_search_indicators)
    
    # Force search for questions about current events
    current_event_patterns = [
        r"\b(what|who|when|where|how).*(today|now|currently|latest|recent)",
        r"\b(current|latest|recent|new).*(news|events|updates)",
        r"\b(price|cost|value).*(today|now|current)",
        r"\b(weather|temperature).*(today|now|current)"
    ]
    
    has_current_event = any(re.search(pattern, message_lower) for pattern in current_event_patterns)
    
    return has_search_trigger or has_current_event

def perform_duckduckgo_search(query, num_results=5):
    """Perform web search using DuckDuckGo instant answer API and web scraping"""
    try:
        search_results = []
        
        # First try DuckDuckGo Instant Answer API
        instant_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1"
        
        response = requests.get(instant_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract instant answer if available
        if data.get('AbstractText'):
            search_results.append({
                'title': data.get('Heading', 'DuckDuckGo Instant Answer'),
                'snippet': data.get('AbstractText'),
                'link': data.get('AbstractURL', ''),
                'source': 'DuckDuckGo Instant Answer'
            })
        
        # If we have related topics, add them
        if data.get('RelatedTopics'):
            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    search_results.append({
                        'title': topic.get('Text', '').split(' - ')[0],
                        'snippet': topic.get('Text', ''),
                        'link': topic.get('FirstURL', ''),
                        'source': 'DuckDuckGo Related'
                    })
        
        # If we don't have enough results, try web scraping
        if len(search_results) < 3:
            scrape_results = scrape_duckduckgo_web(query, num_results - len(search_results))
            search_results.extend(scrape_results)
        
        return search_results[:num_results]
        
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []

def scrape_duckduckgo_web(query, num_results=3):
    """Scrape DuckDuckGo web search results"""
    try:
        # Add random delay to avoid rate limiting
        time.sleep(0.5)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []
        
        # Find search result containers
        for result in soup.find_all('div', class_='result')[:num_results]:
            try:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    
                    # Clean up the link (DuckDuckGo sometimes uses redirects)
                    if link.startswith('/l/?uddg='):
                        link = link.split('uddg=')[1] if 'uddg=' in link else link
                    
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and len(title) > 0:
                        results.append({
                            'title': title,
                            'snippet': snippet,
                            'link': link,
                            'source': 'DuckDuckGo Web'
                        })
            except Exception as e:
                continue
        
        return results
        
    except Exception as e:
        print(f"Web scraping error: {e}")
        return []

def format_search_results(results):
    """Format search results for AI consumption"""
    if not results:
        return ""
    
    formatted = "\n\n--- WEB SEARCH RESULTS ---\n"
    for i, result in enumerate(results, 1):
        formatted += f"{i}. {result['title']}\n"
        if result.get('link'):
            formatted += f"   Source: {result['link']}\n"
        formatted += f"   {result['snippet']}\n\n"
    
    formatted += "--- END SEARCH RESULTS ---\n"
    return formatted

def estimate_tokens(text):
    """Estimate token count from text length"""
    if isinstance(text, list):
        total_chars = 0
        for item in text:
            if isinstance(item, dict) and item.get("type") == "text":
                total_chars += len(item.get("text", ""))
        return total_chars // TOKENS_PER_CHAR
    return len(str(text)) // TOKENS_PER_CHAR

def get_limited_history(conversation_id, max_tokens=MAX_CONTEXT_TOKENS):
    """Get conversation history limited by token count"""
    history = chat_histories.get(conversation_id, [])
    if not history:
        return []
    
    messages = []
    total_tokens = 0
    
    # Work backwards from most recent messages
    for msg in reversed(history):
        if msg["role"] in ["user", "assistant"]:
            # Estimate tokens for this message
            content = msg["content"]
            if isinstance(content, list):
                text_content = next((item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"), "")
                msg_tokens = estimate_tokens(text_content) + 100
            else:
                msg_tokens = estimate_tokens(content)
            
            # If adding this message would exceed limit, stop
            if total_tokens + msg_tokens > max_tokens and messages:
                break
                
            # Add message to front of list (since we're working backwards)
            if msg["role"] == "user" and isinstance(msg["content"], list):
                text_content = next((item.get("text", "") for item in msg["content"] if isinstance(item, dict) and item.get("type") == "text"), "")
                if text_content:
                    messages.insert(0, {"role": "user", "content": text_content})
            else:
                content = msg["content"]
                if isinstance(content, list):
                    content = next((item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"), str(content))
                messages.insert(0, {"role": msg["role"], "content": content})
            
            total_tokens += msg_tokens
    
    return messages

def get_timestamp():
    return datetime.now().isoformat()

def extract_file_content(uploaded_file):
    try:
        filename = uploaded_file.filename.lower()
        file_content = uploaded_file.read()
        uploaded_file.seek(0)
        
        if filename.endswith('.pdf'):
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                return "\n".join(page.extract_text() for page in pdf_reader.pages).strip()
            except Exception as e:
                return f"Error reading PDF: {str(e)}"
        
        elif filename.endswith('.docx'):
            try:
                doc = docx.Document(BytesIO(file_content))
                return "\n".join(p.text for p in doc.paragraphs).strip()
            except Exception as e:
                return f"Error reading Word document: {str(e)}"
        
        elif filename.endswith('.csv'):
            try:
                df = pd.read_csv(BytesIO(file_content))
                return f"CSV File Analysis:\nColumns: {', '.join(df.columns.tolist())}\nShape: {df.shape[0]} rows, {df.shape[1]} columns\n\nFirst 5 rows:\n{df.head().to_string()}"
            except Exception as e:
                return f"Error reading CSV: {str(e)}"
        
        elif filename.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(BytesIO(file_content))
                return f"Excel File Analysis:\nColumns: {', '.join(df.columns.tolist())}\nShape: {df.shape[0]} rows, {df.shape[1]} columns\n\nFirst 5 rows:\n{df.head().to_string()}"
            except Exception as e:
                return f"Error reading Excel file: {str(e)}"
        
        elif filename.endswith(('.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml')):
            try:
                return file_content.decode('utf-8')
            except:
                try:
                    return file_content.decode('latin-1')
                except:
                    return "Error: Could not decode text file"
        
        elif filename.endswith('.json'):
            try:
                return f"JSON File Contents:\n{json.dumps(json.loads(file_content.decode('utf-8')), indent=2)}"
            except Exception as e:
                return f"Error reading JSON: {str(e)}"
        
        elif filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            return None
        
        else:
            return f"Unsupported file type: {filename}"
            
    except Exception as e:
        return f"Error processing file: {str(e)}"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.form.get("message", "")
        selected_model = request.form.get("model", "gpt-4o")
        mode = request.form.get("mode", "general")
        uploaded_files = request.files.getlist("files")
        conversation_id = request.form.get("conversation_id", "default")
        
        if conversation_id not in chat_histories:
            chat_histories[conversation_id] = []
            conversation_metadata[conversation_id] = {
                "created": get_timestamp(),
                "last_active": get_timestamp(),
                "title": "New Chat",
                "message_count": 0
            }
        
        conversation_metadata[conversation_id]["last_active"] = get_timestamp()
        conversation_metadata[conversation_id]["message_count"] += 1
        
        # Check if we should perform web search
        search_results = []
        if user_message and should_search(user_message, mode):
            search_results = perform_duckduckgo_search(user_message)
        
        return handle_multiple_files(user_message, selected_model, mode, uploaded_files, conversation_id, search_results) if uploaded_files else handle_text_chat(user_message, selected_model, mode, conversation_id, search_results)
            
    except Exception as e:
        print("Error occurred:", traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def handle_text_chat(user_message, selected_model, mode, conversation_id, search_results=None):
    try:
        # Handle /help command
        if user_message.strip().lower() == "/help":
            help_text = """🤖 Chat Modes Help

Available Modes:

💬 General Mode (Default)
- No special prompting or instructions
- Natural conversation with base AI
- Good for: General questions, casual chat, mixed topics

🔧 Coding Mode
- Expert software engineer with lower temperature (0.1)
- Focuses on: Clean code, best practices, debugging, explanations
- Good for: Programming, code review, technical implementation

🎨 Creative Mode
- Creative writing assistant with higher temperature (0.8)
- Focuses on: Storytelling, brainstorming, artistic expression
- Good for: Writing stories, generating ideas, creative projects

📊 Analysis Mode
- Data analyst with structured thinking (temperature 0.2)
- Focuses on: Logical breakdown, data-driven reasoning, insights
- Good for: Research, problem-solving, systematic analysis

📚 Learning Mode
- Patient tutor with balanced temperature (0.3)
- Focuses on: Step-by-step teaching, examples, clear explanations
- Good for: Learning concepts, studying, educational content

🦆 Web Search Features:
- Automatic search for current events, news, prices
- Manual search with keywords like "search", "find", "latest"
- Powered by DuckDuckGo (free, no API key required)

Tips:
- Switch modes anytime using the dropdown
- Each mode optimizes temperature and prompting for its specialty
- General mode = pure AI with no special instructions
- Type /help anytime to see this guide"""
            
            chat_histories[conversation_id].extend([
                {"role": "user", "content": user_message, "timestamp": get_timestamp()},
                {"role": "assistant", "content": help_text, "timestamp": get_timestamp()}
            ])
            
            return jsonify({"response": help_text})
        
        # Prepare message with search results if available
        enhanced_message = user_message
        if search_results:
            search_context = format_search_results(search_results)
            enhanced_message = f"{user_message}\n{search_context}"
        
        messages = get_limited_history(conversation_id, MAX_CONTEXT_TOKENS - 500)
        
        # Add system prompt based on mode
        system_prompt = get_system_prompt(mode, bool(search_results))
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": enhanced_message})
        
        # Count input tokens before sending
        input_tokens = sum(count_tokens_tiktoken(msg["content"], selected_model) for msg in messages)
        
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=get_temperature_for_mode(mode)
        )
        
        assistant_response = response.choices[0].message.content
        
        # Count output tokens and update usage
        output_tokens = count_tokens_tiktoken(assistant_response, selected_model)
        token_counter = update_token_usage(input_tokens, output_tokens, selected_model)
        
        # Add search indicator to response if search was used
        if search_results:
            assistant_response += f"\n\n*🦆 Used DuckDuckGo search ({len(search_results)} results)*"
        
        chat_histories[conversation_id].extend([
            {"role": "user", "content": user_message, "timestamp": get_timestamp(), "search_results": search_results if search_results else None},
            {"role": "assistant", "content": assistant_response, "timestamp": get_timestamp()}
        ])
        
        return jsonify({
            "response": assistant_response,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "daily_total": token_counter["total_tokens"],
                "daily_cost": token_counter["cost_estimate"]
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Text processing failed: {str(e)}"}), 500

def handle_multiple_files(user_message, selected_model, mode, uploaded_files, conversation_id, search_results=None):
    try:
        image_files = []
        document_files = []
        
        for file in uploaded_files:
            if any(file.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
                image_files.append(file)
            else:
                document_files.append(file)
        
        document_contents = []
        for doc_file in document_files:
            content = extract_file_content(doc_file)
            if content is not None:
                document_contents.append(f"--- {doc_file.filename} ---\n{content}")
        
        all_document_content = "\n\n".join(document_contents) if document_contents else ""
        
        if len(all_document_content) > 12000:
            all_document_content = all_document_content[:12000] + "\n\n[Content truncated due to length...]"
        
        # Add search results if available
        if search_results:
            search_context = format_search_results(search_results)
            all_document_content += f"\n\n{search_context}"
        
        if user_message:
            combined_message = f"{user_message}\n\nDocument contents:\n{all_document_content}" if all_document_content else user_message
        else:
            combined_message = f"Please analyze these files:\n\n{all_document_content}" if all_document_content else "Please analyze the uploaded files."
        
        return handle_multiple_images_with_documents(combined_message, selected_model, mode, image_files, conversation_id, [f.filename for f in uploaded_files], search_results) if image_files else handle_text_chat_with_content(combined_message, selected_model, mode, conversation_id, [f.filename for f in uploaded_files], user_message, search_results)
        
    except Exception as e:
        return jsonify({"error": f"File processing failed: {str(e)}"}), 500

def handle_multiple_images_with_documents(message_content, selected_model, mode, image_files, conversation_id, all_filenames, search_results=None):
    try:
        messages = get_limited_history(conversation_id, MAX_CONTEXT_TOKENS - 1000)  # Reserve more space for images
        
        # Add system prompt based on mode
        system_prompt = get_system_prompt(mode, bool(search_results))
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        image_data_urls = []
        for image_file in image_files:
            file_content = image_file.read()
            image_file.seek(0)
            base64_image = base64.b64encode(file_content).decode('utf-8')
            
            filename = image_file.filename.lower()
            mime_type = 'image/png' if filename.endswith('.png') else 'image/jpeg' if filename.endswith(('.jpg', '.jpeg')) else 'image/gif' if filename.endswith('.gif') else 'image/webp' if filename.endswith('.webp') else 'image/jpeg'
            
            image_data_urls.append(f"data:{mime_type};base64,{base64_image}")
        
        message_content_array = [{"type": "text", "text": message_content}]
        message_content_array.extend([{"type": "image_url", "image_url": {"url": url}} for url in image_data_urls])
        
        current_message = {"role": "user", "content": message_content_array}
        messages.append(current_message)
        
        # Count input tokens (text only for now, images are complex)
        text_messages = []
        for msg in messages:
            if isinstance(msg["content"], list):
                text_content = next((item.get("text", "") for item in msg["content"] if isinstance(item, dict) and item.get("type") == "text"), "")
                if text_content:
                    text_messages.append({"role": msg["role"], "content": text_content})
            else:
                text_messages.append(msg)
        
        input_tokens = sum(count_tokens_tiktoken(msg["content"], selected_model) for msg in text_messages)
        
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=get_temperature_for_mode(mode)
        )
        
        assistant_response = response.choices[0].message.content
        
        # Count output tokens and update usage
        output_tokens = count_tokens_tiktoken(assistant_response, selected_model)
        token_counter = update_token_usage(input_tokens, output_tokens, selected_model)
        
        # Add search indicator if search was used
        if search_results:
            assistant_response += f"\n\n*🦆 Used DuckDuckGo search ({len(search_results)} results)*"
        
        chat_histories[conversation_id].extend([
            {"role": "user", "content": current_message["content"], "file_attachments": all_filenames, "image_data": image_data_urls, "timestamp": get_timestamp(), "search_results": search_results if search_results else None},
            {"role": "assistant", "content": assistant_response, "timestamp": get_timestamp()}
        ])
        
        return jsonify({
            "response": assistant_response,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "daily_total": token_counter["total_tokens"],
                "daily_cost": token_counter["cost_estimate"]
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Multiple image processing failed: {str(e)}"}), 500

def handle_text_chat_with_content(message_content, selected_model, mode, conversation_id, filenames=None, original_user_message=None, search_results=None):
    try:
        messages = get_limited_history(conversation_id, MAX_CONTEXT_TOKENS - 500)
        
        # Add system prompt based on mode
        system_prompt = get_system_prompt(mode, bool(search_results))
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": message_content})
        
        # Count input tokens
        input_tokens = sum(count_tokens_tiktoken(msg["content"], selected_model) for msg in messages)
        
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=get_temperature_for_mode(mode)
        )
        
        assistant_response = response.choices[0].message.content
        
        # Count output tokens and update usage
        output_tokens = count_tokens_tiktoken(assistant_response, selected_model)
        token_counter = update_token_usage(input_tokens, output_tokens, selected_model)
        
        # Add search indicator if search was used
        if search_results:
            assistant_response += f"\n\n*🦆 Used DuckDuckGo search ({len(search_results)} results)*"
        
        if filenames:
            display_message = original_user_message if original_user_message else (f"📎 {filenames[0]}" if len(filenames) == 1 else f"📎 {len(filenames)} files: {', '.join(filenames)}")
            chat_histories[conversation_id].append({"role": "user", "content": display_message, "file_attachments": filenames, "timestamp": get_timestamp(), "search_results": search_results if search_results else None})
        else:
            chat_histories[conversation_id].append({"role": "user", "content": message_content, "timestamp": get_timestamp(), "search_results": search_results if search_results else None})
        
        chat_histories[conversation_id].append({"role": "assistant", "content": assistant_response, "timestamp": get_timestamp()})
        
        return jsonify({
            "response": assistant_response,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "daily_total": token_counter["total_tokens"],
                "daily_cost": token_counter["cost_estimate"]
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Text processing failed: {str(e)}"}), 500

@app.route("/token_usage", methods=["GET"])
def get_token_usage():
    """Get current token usage statistics"""
    try:
        today_counter = get_token_counter()
        
        # Get last 7 days of data
        history = []
        today = date.today()
        for i in range(7):
            check_date = str(today - pd.Timedelta(days=i))
            if check_date in daily_token_usage:
                day_data = daily_token_usage[check_date].copy()
                day_data["date_formatted"] = datetime.strptime(check_date, "%Y-%m-%d").strftime("%b %d")
                history.append(day_data)
            else:
                history.append({
                    "date": check_date,
                    "date_formatted": (today - pd.Timedelta(days=i)).strftime("%b %d"),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "requests": 0,
                    "cost_estimate": 0.0
                })
        
        return jsonify({
            "today": today_counter,
            "history": list(reversed(history)),  # Oldest first
            "total_days_tracked": len(daily_token_usage)
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to get token usage: {str(e)}"}), 500

@app.route("/reset_tokens", methods=["POST"])
def reset_daily_tokens():
    """Manually reset today's token counter"""
    try:
        today = str(date.today())
        if today in daily_token_usage:
            del daily_token_usage[today]
        
        new_counter = get_token_counter()
        return jsonify({
            "success": True,
            "message": "Today's token counter has been reset",
            "counter": new_counter
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to reset tokens: {str(e)}"}), 500

@app.route("/search", methods=["POST"])
def manual_search():
    """Manual search endpoint for explicit search requests"""
    try:
        data = request.json
        query = data.get("query", "")
        num_results = data.get("num_results", 5)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        results = perform_duckduckgo_search(query, num_results)
        return jsonify({"results": results, "query": query})
        
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        conversation_id = request.json.get("conversation_id", "default")
        
        for storage in [chat_histories, conversation_metadata]:
            if conversation_id in storage:
                del storage[conversation_id]
        
        return jsonify({"success": True, "message": "History cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_history", methods=["GET"])
def get_history():
    try:
        conversation_id = request.args.get("conversation_id", "default")
        return jsonify({"history": chat_histories.get(conversation_id, [])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_conversations", methods=["GET"])
def get_conversations():
    try:
        conversations = [{"id": conv_id, **metadata} for conv_id, metadata in conversation_metadata.items()]
        conversations.sort(key=lambda x: x["last_active"], reverse=True)
        return jsonify({"conversations": conversations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update_conversation", methods=["POST"])
def update_conversation():
    try:
        data = request.json
        conversation_id = data.get("conversation_id")
        new_title = data.get("title")
        
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        
        if conversation_id not in conversation_metadata:
            conversation_metadata[conversation_id] = {"created": get_timestamp(), "last_active": get_timestamp(), "title": "New Chat", "message_count": 0}
        
        if new_title:
            conversation_metadata[conversation_id]["title"] = new_title
        
        conversation_metadata[conversation_id]["last_active"] = get_timestamp()
        
        return jsonify({"success": True, "message": "Conversation updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete_conversation", methods=["POST"])
def delete_conversation():
    try:
        conversation_id = request.json.get("conversation_id")
        
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        
        if len(conversation_metadata) <= 1:
            return jsonify({"error": "Cannot delete the last conversation"}), 400
        
        for storage in [chat_histories, conversation_metadata]:
            if conversation_id in storage:
                del storage[conversation_id]
        
        return jsonify({"success": True, "message": "Conversation deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    today_counter = get_token_counter()
    return jsonify({
        "status": "healthy", 
        "timestamp": get_timestamp(), 
        "active_conversations": len(conversation_metadata),
        "search_enabled": True,
        "search_provider": "DuckDuckGo",
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "response_token_limit": "Unlimited",
        "daily_tokens": today_counter["total_tokens"],
        "daily_cost": f"${today_counter['cost_estimate']:.4f}",
        "daily_requests": today_counter["requests"]
    })

if __name__ == "__main__":
    print("🦆 DuckDuckGo web search enabled (free)")
    print(f"📝 Context limit: {MAX_CONTEXT_TOKENS} tokens")
    print("🚀 Response limit: UNLIMITED")
    print("📊 Token tracking enabled with daily reset")
    
    # Install tiktoken if not available
    try:
        import tiktoken
        print("✅ tiktoken found - accurate token counting enabled")
    except ImportError:
        print("⚠️  tiktoken not found - install with: pip install tiktoken")
        print("   Using fallback character-based estimation")
    
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)

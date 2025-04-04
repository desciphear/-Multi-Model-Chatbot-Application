import streamlit as st
import time
from openai import OpenAI
import base64
import json
import zipfile
import io
from PIL import Image
import requests
import PyPDF2

# Model dictionary with display names as keys and model IDs as values (alphabetically ordered)
MODEL_DICT = {
    "DeepHermes": "nousresearch/deephermes-3-llama-3-8b-preview:free",
    "DeepSeek Chat": "deepseek/deepseek-chat-v3-0324:free",
    "DeepSeek Chat Base": "deepseek/deepseek-chat:free",
    "DeepSeek LLaMA 70B": "deepseek/deepseek-r1-distill-llama-70b:free",
    "DeepSeek Qwen 14B": "deepseek/deepseek-r1-distill-qwen-14b:free",
    "DeepSeek Qwen 32B": "deepseek/deepseek-r1-distill-qwen-32b:free",
    "DeepSeek R1": "deepseek/deepseek-r1:free",
    "Dolphin 3.0": "cognitivecomputations/dolphin3.0-mistral-24b:free",
    "Dolphin 3.0 R1": "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "Gemini Flash": "google/gemini-flash-1.5-8b-exp",
    "Gemini Flash Lite": "google/gemini-2.0-flash-lite-preview-02-05:free",
    "Gemini Flash Think": "google/gemini-2.0-flash-thinking-exp:free",
    "Gemini Pro": "google/gemini-2.0-pro-exp-02-05:free",
    "Gemma 1B": "google/gemma-3-1b-it:free",
    "Gemma 4B": "google/gemma-3-4b-it:free",
    "Gemma 9B": "google/gemma-2-9b-it:free",
    "Gemma 12B": "google/gemma-3-12b-it:free",
    "Gemma 27B": "google/gemma-3-27b-it:free",
    "LearnLM Pro": "google/learnlm-1.5-pro-experimental:free",
    "LLaMA 3 8B": "meta-llama/llama-3-8b-instruct:free",
    "LLaMA 3.1 8B": "meta-llama/llama-3.1-8b-instruct:free",
    "LLaMA 3.2 1B": "meta-llama/llama-3.2-1b-instruct:free",
    "LLaMA 3.2 3B": "meta-llama/llama-3.2-3b-instruct:free",
    "LLaMA 3.3 70B": "meta-llama/llama-3.3-70b-instruct:free",
    "LLaMA Vision": "meta-llama/llama-3.2-11b-vision-instruct:free",
    "Mistral 7B": "mistralai/mistral-7b-instruct:free",
    "Mistral Nemo": "mistralai/mistral-nemo:free",
    "Mistral Small": "mistralai/mistral-small-3.1-24b-instruct:free",
    "Moonlight 16B": "moonshotai/moonlight-16b-a3b-instruct:free",
    "MythoMax": "gryphe/mythomax-l2-13b:free",
    "Nemotron 70B": "nvidia/llama-3.1-nemotron-70b-instruct:free",
    "Olympic Coder 7B": "open-r1/olympiccoder-7b:free",
    "Olympic Coder 32B": "open-r1/olympiccoder-32b:free",
    "OpenChat": "openchat/openchat-7b:free",
    "Phi-3 Medium": "microsoft/phi-3-medium-128k-instruct:free",
    "Phi-3 Mini": "microsoft/phi-3-mini-128k-instruct:free",
    "Qwen 7B": "qwen/qwen-2-7b-instruct:free",
    "Qwen 72B": "qwen/qwen2.5-vl-72b-instruct:free",
    "Qwen 72B Base": "qwen/qwen-2.5-72b-instruct:free",
    "Qwen 2.5": "qwen/qwen2.5-vl-32b-instruct:free",
    "Qwen Coder": "qwen/qwen-2.5-coder-32b-instruct:free",
    "QWQ 32B": "qwen/qwq-32b-preview:free",
    "Reka Flash": "rekaai/reka-flash-3:free",
    "Rogue Rose": "sophosympatheia/rogue-rose-103b-v0.2:free",
    "Toppy M": "undi95/toppy-m-7b:free",
    "Zephyr": "huggingfaceh4/zephyr-7b-beta:free"
}

# First, add a new dictionary with model information
MODEL_INFO = {
    "DeepHermes": {
        "parameters": "3.8B",
        "context_length": "8K",
        "architecture": "LLaMA-based",
        "description": "A compact yet powerful model optimized for instruction following and coding tasks.",
        "best_at": "Code generation, general knowledge tasks, and instruction following",
        "release_date": "January 15, 2024"
    },
    "DeepSeek Chat": {
        "parameters": "33B",
        "context_length": "32K",
        "architecture": "Transformer",
        "description": "Specialized chat model with strong dialogue capabilities and knowledge base.",
        "best_at": "Open-domain dialogue, task completion, and knowledge retrieval",
        "release_date": "October 17, 2023"
    },
    "DeepSeek Chat Base": {
        "parameters": "16B",
        "context_length": "16K",
        "architecture": "Transformer",
        "description": "Foundation model for chat applications with balanced performance.",
        "best_at": "General conversation and basic task completion",
        "release_date": "September 12, 2023"
    },  
    "DeepSeek LLaMA 70B": {
        "parameters": "70B",
        "context_length": "32K",
        "architecture": "LLaMA-based",
        "description": "Large-scale model with comprehensive capabilities and deep understanding.",
        "best_at": "Complex reasoning, analysis, and generation tasks",
        "release_date": "December 5, 2023"
    },
    "DeepSeek Qwen 14B": {
        "parameters": "14B",
        "context_length": "16K",
        "architecture": "Qwen-based",
        "description": "Efficient model combining DeepSeek and Qwen architectures.",
        "best_at": "Balanced performance across various tasks",
        "release_date": "January 20, 2024"
    },
    "DeepSeek Qwen 32B": {
        "parameters": "32B",
        "context_length": "32K",
        "architecture": "Qwen-based",
        "description": "Larger Qwen variant with enhanced capabilities.",
        "best_at": "Complex reasoning and generation tasks",
        "release_date": "January 20, 2024"
    },
    "DeepSeek R1": {
        "parameters": "176B",
        "context_length": "32K",
        "architecture": "Custom Transformer",
        "description": "Advanced research model with state-of-the-art capabilities.",
        "best_at": "Research-oriented tasks and complex problem-solving",
        "release_date": "February 1, 2024"
    },
    "Dolphin 3.0": {
        "parameters": "24B",
        "context_length": "16K",
        "architecture": "Mistral-based",
        "description": "Enhanced Mistral model with improved instruction following.",
        "best_at": "General tasks and instruction following",
        "release_date": "December 15, 2023"
    },
    "Dolphin 3.0 R1": {
        "parameters": "24B",
        "context_length": "16K",
        "architecture": "Mistral-based",
        "description": "Refined version of Dolphin 3.0 with better performance.",
        "best_at": "Instruction following and general tasks",
        "release_date": "December 15, 2023"
    },
    "Gemini Flash": {
        "parameters": "8B",
        "context_length": "16K",
        "architecture": "Google Transformer",
        "description": "Fast and efficient model for quick responses.",
        "best_at": "Real-time applications and quick responses",
        "release_date": "December 6, 2023"
    },
    "Gemini Flash Lite": {
        "parameters": "4B",
        "context_length": "8K",
        "architecture": "Google Transformer",
        "description": "Lightweight version of Gemini Flash for mobile and edge devices.",
        "best_at": "Mobile and edge computing applications",
        "release_date": "December 6, 2023"
    },
    "Gemini Flash Think": {
        "parameters": "8B",
        "context_length": "16K",
        "architecture": "Google Transformer",
        "description": "Enhanced reasoning capabilities with Flash architecture.",
        "best_at": "Quick reasoning and analysis tasks",
        "release_date": "December 6, 2023"
    },
    "Gemini Pro": {
        "parameters": "Unknown",
        "context_length": "32K",
        "architecture": "Google Transformer",
        "description": "Google's advanced language model with strong multi-task capabilities.",
        "best_at": "General knowledge, reasoning, and creative tasks",
        "release_date": "December 6, 2023"
    },
    "Gemma 1B": {
        "parameters": "1B",
        "context_length": "8K",
        "architecture": "Gemma",
        "description": "Compact model for basic tasks and edge devices.",
        "best_at": "Basic language tasks and edge deployment",
        "release_date": "February 21, 2024"
    },
    "Gemma 4B": {
        "parameters": "4B",
        "context_length": "8K",
        "architecture": "Gemma",
        "description": "Balanced model for general-purpose applications.",
        "best_at": "General language tasks and basic reasoning",
        "release_date": "February 21, 2024"
    },
    "Gemma 9B": {
        "parameters": "9B",
        "context_length": "16K",
        "architecture": "Gemma",
        "description": "Mid-sized model with improved capabilities.",
        "best_at": "Complex language tasks and reasoning",
        "release_date": "February 21, 2024"
    },
    "Gemma 12B": {
        "parameters": "12B",
        "context_length": "16K",
        "architecture": "Gemma",
        "description": "Advanced model with strong performance.",
        "best_at": "Complex reasoning and generation tasks",
        "release_date": "February 21, 2024"
    },
    "Gemma 27B": {
        "parameters": "27B",
        "context_length": "32K",
        "architecture": "Gemma",
        "description": "Large Gemma model with comprehensive capabilities.",
        "best_at": "Advanced reasoning and complex tasks",
        "release_date": "February 21, 2024"
    },
    "LearnLM Pro": {
        "parameters": "Unknown",
        "context_length": "32K",
        "architecture": "Custom Transformer",
        "description": "Specialized model for educational applications.",
        "best_at": "Educational content and tutoring",
        "release_date": "March 1, 2024"
    },
    "LLaMA 3 8B": {
        "parameters": "8B",
        "context_length": "16K",
        "architecture": "LLaMA 3",
        "description": "Latest generation of Meta's LLaMA architecture.",
        "best_at": "General language tasks and instruction following",
        "release_date": "November 2023 - March 2024"
    },
    "LLaMA 3.1 8B": {
        "parameters": "8B",
        "context_length": "16K",
        "architecture": "LLaMA 3.1",
        "description": "Improved version of LLaMA 3 with better performance.",
        "best_at": "Enhanced language understanding and generation",
        "release_date": "November 2023 - March 2024"
    },
    "LLaMA 3.2 1B": {
        "parameters": "1B",
        "context_length": "8K",
        "architecture": "LLaMA 3.2",
        "description": "Compact version of LLaMA 3.2 for efficient deployment.",
        "best_at": "Basic language tasks and edge computing",
        "release_date": "November 2023 - March 2024"
    },
    "LLaMA 3.2 3B": {
        "parameters": "3B",
        "context_length": "8K",
        "architecture": "LLaMA 3.2",
        "description": "Mid-sized LLaMA 3.2 model with balanced performance.",
        "best_at": "General language tasks and basic reasoning",
        "release_date": "November 2023 - March 2024"
    },
    "LLaMA 3.3 70B": {
        "parameters": "70B",
        "context_length": "32K",
        "architecture": "LLaMA 3.3",
        "description": "Large-scale model with state-of-the-art performance.",
        "best_at": "Complex reasoning and advanced tasks",
        "release_date": "November 2023 - March 2024"
    },
    "LLaMA Vision": {
        "parameters": "11B",
        "context_length": "32K",
        "architecture": "LLaMA Vision",
        "description": "Multimodal model supporting both text and image inputs.",
        "best_at": "Vision-language tasks and image understanding",
        "release_date": "February 15, 2024"
    },
    "Mistral 7B": {
        "parameters": "7B",
        "context_length": "8K",
        "architecture": "Mistral",
        "description": "Efficient base model with strong performance.",
        "best_at": "General purpose tasks and instruction following",
        "release_date": "September 2023 - February 2024"
    },
    "Mistral Nemo": {
        "parameters": "7B",
        "context_length": "32K",
        "architecture": "Mistral",
        "description": "Enhanced version of Mistral with extended context.",
        "best_at": "Long-form content and complex reasoning",
        "release_date": "September 2023 - February 2024"
    },
    "Mistral Small": {
        "parameters": "24B",
        "context_length": "32K",
        "architecture": "Mistral-based",
        "description": "Powerful instruction-tuned model with strong reasoning.",
        "best_at": "Complex reasoning and detailed explanations",
        "release_date": "September 2023 - February 2024"
    },
    "Moonlight 16B": {
        "parameters": "16B",
        "context_length": "16K",
        "architecture": "Custom Transformer",
        "description": "Specialized model for creative and analytical tasks.",
        "best_at": "Creative writing and analysis",
        "release_date": "January 10, 2024"
    },
    "MythoMax": {
        "parameters": "13B",
        "context_length": "16K",
        "architecture": "Custom",
        "description": "Model focused on creative and narrative generation.",
        "best_at": "Storytelling and creative writing",
        "release_date": "December 20, 2023"
    },
    "Nemotron 70B": {
        "parameters": "70B",
        "context_length": "32K",
        "architecture": "NVIDIA Custom",
        "description": "NVIDIA's large-scale model with comprehensive capabilities.",
        "best_at": "Complex reasoning and advanced tasks",
        "release_date": "February 28, 2024"
    },
    "Olympic Coder 7B": {
        "parameters": "7B",
        "context_length": "16K",
        "architecture": "Custom",
        "description": "Specialized model for code generation and understanding.",
        "best_at": "Programming and code-related tasks",
        "release_date": "January 25, 2024"
    },
    "Olympic Coder 32B": {
        "parameters": "32B",
        "context_length": "32K",
        "architecture": "Custom",
        "description": "Large-scale coding model with advanced capabilities.",
        "best_at": "Complex programming and software development",
        "release_date": "January 25, 2024"
    },
    "OpenChat": {
        "parameters": "7B",
        "context_length": "16K",
        "architecture": "Custom",
        "description": "Open-source chat model with strong dialogue capabilities.",
        "best_at": "Conversational tasks and chat applications",
        "release_date": "October 2023"
    },
    "Phi-3 Medium": {
        "parameters": "8B",
        "context_length": "128K",
        "architecture": "Microsoft Phi",
        "description": "Microsoft's efficient model with very long context.",
        "best_at": "Long-form content processing",
        "release_date": "March 1, 2024"
    },
    "Phi-3 Mini": {
        "parameters": "3B",
        "context_length": "128K",
        "architecture": "Microsoft Phi",
        "description": "Compact version of Phi-3 with long context support.",
        "best_at": "Efficient long-form processing",
        "release_date": "March 1, 2024"
    },
    "Qwen 7B": {
        "parameters": "7B",
        "context_length": "16K",
        "architecture": "Qwen",
        "description": "Efficient base model with balanced capabilities.",
        "best_at": "General language tasks",
        "release_date": "September 2023 - February 2024"
    },
    "Qwen 72B": {
        "parameters": "72B",
        "context_length": "32K",
        "architecture": "Qwen",
        "description": "Large-scale model with vision-language capabilities.",
        "best_at": "Multimodal tasks and complex reasoning",
        "release_date": "September 2023 - February 2024"
    },
    "Qwen 72B Base": {
        "parameters": "72B",
        "context_length": "32K",
        "architecture": "Qwen",
        "description": "Base version of the 72B model without vision features.",
        "best_at": "Complex language tasks and reasoning",
        "release_date": "September 2023 - February 2024"
    },
    "Qwen 2.5": {
        "parameters": "32B",
        "context_length": "32K",
        "architecture": "Qwen",
        "description": "Latest generation of Qwen with improved capabilities.",
        "best_at": "Advanced language tasks and reasoning",
        "release_date": "September 2023 - February 2024"
    },
    "Qwen Coder": {
        "parameters": "32B",
        "context_length": "32K",
        "architecture": "Qwen",
        "description": "Specialized coding variant of Qwen.",
        "best_at": "Programming and software development",
        "release_date": "September 2023 - February 2024"
    },
    "QWQ 32B": {
        "parameters": "32B",
        "context_length": "32K",
        "architecture": "Custom",
        "description": "Experimental model with novel architecture.",
        "best_at": "Research and experimental applications",
        "release_date": "March 1, 2024"
    },
    "Reka Flash": {
        "parameters": "3B",
        "context_length": "8K",
        "architecture": "Custom",
        "description": "Fast and efficient model for quick responses.",
        "best_at": "Real-time applications",
        "release_date": "February 15, 2024"
    },
    "Rogue Rose": {
        "parameters": "103B",
        "context_length": "32K",
        "architecture": "Custom",
        "description": "Large-scale model with unique capabilities.",
        "best_at": "Creative and analytical tasks",
        "release_date": "March 10, 2024"
    },
    "Toppy M": {
        "parameters": "7B",
        "context_length": "16K",
        "architecture": "Custom",
        "description": "Efficient model with balanced performance.",
        "best_at": "General language tasks",
        "release_date": "January 5, 2024"
    },
    "Zephyr": {
        "parameters": "7B",
        "context_length": "16K",
        "architecture": "Custom",
        "description": "Refined model with strong instruction following.",
        "best_at": "Instruction following and general tasks",
        "release_date": "November 15, 2023"
    }
}

# Function to encode files to base64
def encode_file(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        base64_content = base64.b64encode(uploaded_file.getvalue()).decode()
        
        if file_extension in ['png', 'jpg', 'jpeg', 'gif']:
            return {
                "type": "image",
                "content": base64_content,
                "format": file_extension
            }
        elif file_extension == 'zip':
            extracted_files = {}
            with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as z:
                for filename in z.namelist():
                    with z.open(filename) as f:
                        content = base64.b64encode(f.read()).decode()
                        extracted_files[filename] = content
            return {
                "type": "zip",
                "content": extracted_files
            }
        else:
            return {
                "type": "file",
                "content": base64_content,
                "format": file_extension
            }

# Add these functions after the existing encode_file function
def process_text_file(content):
    """Process text file content."""
    try:
        return content.decode('utf-8')
    except UnicodeDecodeError:
        return None

def process_image_file(content):
    """Process image file and return description."""
    try:
        image = Image.open(io.BytesIO(content))
        return f"Image dimensions: {image.size}, Mode: {image.mode}"
    except:
        return None

def extract_file_content(file):
    """Extract and process file content based on type."""
    content = file.read()
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            return text_content
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    if file_extension in ['txt', 'py', 'json', 'md']:
        return process_text_file(content)
    elif file_extension == 'pdf':
        return f"PDF file detected: {file.name}"
    return None

# Function to process and display images from model response
def process_image_response(response_text):
    # Check if response contains base64 image data
    if "data:image" in response_text:
        try:
            # Extract base64 image data
            img_data = response_text.split("data:image")[1].split(";base64,")[1].split("```")[0].strip()
            image_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Display image
            st.image(image, caption="Generated Image")
            
            # Add download button
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="generated_image.png",
                mime="image/png"
            )
            
            # Return cleaned text without base64 data
            return response_text.split("data:image")[0] + response_text.split("```")[1]
        except:
            return response_text
    return response_text

# Add this after the imports
def process_file_uploads():
    """Process and store uploaded files in session state."""
    if "uploaded_file_contents" not in st.session_state:
        st.session_state.uploaded_file_contents = {}
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_file_contents:
                processed_content = extract_file_content(file)
                if processed_content:
                    st.session_state.uploaded_file_contents[file.name] = {
                        "content": processed_content,
                        "type": file.type
                    }
                file.seek(0)

# Add this function to manage file removal
def cleanup_removed_files():
    """Remove file contents from session state when files are removed from uploader."""
    if "uploaded_file_contents" in st.session_state:
        current_files = set(f.name for f in uploaded_files) if uploaded_files else set()
        stored_files = set(st.session_state.uploaded_file_contents.keys())
        removed_files = stored_files - current_files
        
        for file in removed_files:
            del st.session_state.uploaded_file_contents[file]

# Initialize session state variables at the start of your script, after imports
if 'model_messages' not in st.session_state:
    st.session_state.model_messages = {model_id: [] for model_id in MODEL_DICT.values()}

if 'uploaded_file_contents' not in st.session_state:
    st.session_state.uploaded_file_contents = {}

# Add sidebar
with st.sidebar:
    st.title("Chatbot Settings")
    selected_display_name = st.selectbox(
        "Choose Model",
        list(MODEL_DICT.keys()),
        index=0,
        key="model_selector"
    )
    selected_model = MODEL_DICT[selected_display_name]
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    # Add clear button for current model
    if st.button(f"Clear {selected_display_name} Chat"):
        if selected_model in st.session_state.model_messages:
            st.session_state.model_messages[selected_model] = []
            st.rerun()
    
    st.divider()
    st.markdown("### About Model")
    
    # Display model information if available
    if selected_display_name in MODEL_INFO:
        info = MODEL_INFO[selected_display_name]
        st.markdown(f"**Model Name:** {selected_display_name}")
        st.markdown(f"**Parameters:** {info['parameters']}")
        st.markdown(f"**Context Length:** {info['context_length']}")
        st.markdown(f"**Architecture:** {info['architecture']}")
        st.markdown("**Description:**")
        st.markdown(info['description'])
        st.markdown("**Best At:**")
        st.markdown(info['best_at'])
        st.markdown(f"**Release Date:** {info['release_date']}")
    else:
        st.markdown("Detailed information not available for this model.")
    
    # Add file upload section
    st.divider()
    st.markdown("### Upload Files")
    uploaded_files = st.file_uploader(
        "Upload documents, images, code, or zip files",
        type=['txt', 'py', 'json', 'pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        process_file_uploads()
        cleanup_removed_files()  # Add this line to handle file removal
        st.markdown("### Uploaded Files")
        for filename in st.session_state.uploaded_file_contents.keys():
            st.markdown(f"- {filename}")

st.markdown(f"<h1 style='text-align: center; color: black;'>{selected_display_name}</h1>", unsafe_allow_html=True)

def stream_data(inp):
    for word in inp.split(" "):
        yield word + " "
        time.sleep(0.05)

# Initialize message histories for all models
if "model_messages" not in st.session_state:
    st.session_state.model_messages = {model_id: [] for model_id in MODEL_DICT.values()}

# Display messages for the currently selected model
for message in st.session_state.model_messages[selected_model]:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        elif isinstance(message["content"], dict) and "type" in message["content"]:
            if message["content"]["type"] == "image":
                image_bytes = base64.b64decode(message["content"]["content"])
                st.image(image_bytes)

prompt = st.chat_input("Ask Your Question")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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", #It is a free OpenRouter API
)

# Update the chat logic section
if prompt:
    # Prepare message content
    message_content = prompt
    message_for_model = prompt  # Default to just the prompt
    
    # Add file contents only if files are present and explicitly requested in the current prompt
    if st.session_state.uploaded_file_contents and any(
        keyword in prompt.lower() 
        for keyword in ['analyze', 'read', 'show', 'content', 'what is in', 'what does', 'file', 'document']
    ):
        files_content = []
        for filename, file_data in st.session_state.uploaded_file_contents.items():
            files_content.append({
                "filename": filename,
                "content": file_data["content"]
            })
        message_for_model = f"{message_content}\n\nAvailable files:\n{json.dumps(files_content, indent=2)}"
    
    # Display visible message to user (without file contents)
    with st.chat_message("user"):
        st.markdown(message_content)
    
    # Store only the user's message without file contents for history
    st.session_state.model_messages[selected_model].append({
        "role": "user",
        "content": message_content
    })
    
    with st.spinner("Thinking..."):
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.model_messages[selected_model]
            ] + ([{"role": "user", "content": message_for_model}] if message_for_model != message_content else []),
            temperature=temperature
        )
        response = completion.choices[0].message.content
        
        # Process response for images
        response = process_image_response(response)
        
    with st.chat_message("assistant"):
        st.write_stream(stream_data(response))
    
    st.session_state.model_messages[selected_model].append({
        "role": "assistant",
        "content": response
    })







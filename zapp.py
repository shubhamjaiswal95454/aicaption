import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import yake

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_llm(model_size):
    if model_size == "distilgpt2":
        model_name = "distilgpt2"
    elif model_size == "gpt2-medium":
        model_name = "gpt2-medium"
    else:
        model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def extract_keywords_yake(text, num_keywords=5):
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=num_keywords)
    keywords = [kw for kw, _ in kw_extractor.extract_keywords(text)]
    hashtags = []
    for k in keywords:
        tag = "#" + k.replace(" ", "").replace("-", "").replace("_", "")
        if tag.isalnum() or tag.replace("#", "").isalnum():
            hashtags.append(tag.lower())
    return hashtags

def get_llm_caption(tokenizer, model, prompt, max_tokens=25):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.85,
            pad_token_id=tokenizer.eos_token_id
        )
    llm_cap = tokenizer.decode(output[0], skip_special_tokens=True)
    if llm_cap.lower().startswith(prompt.lower()):
        llm_cap = llm_cap[len(prompt):].strip()
    sentence = llm_cap.split(".")[0][:80].strip(" .?!,")
    return sentence if sentence else llm_cap.strip()

st.title("✨ Local AI Social Media Captioner")
st.markdown(
    "Upload a photo and get a **creative, styled, and hashtagged social media caption.** <br>"
    "Runs entirely <b>locally</b> for privacy — free, no API keys!",
    unsafe_allow_html=True
)

llm_choice = st.radio(
    "Choose AI model for creative caption:",
    ["Fast & Lightweight (distilgpt2)", "Richer (gpt2-medium - needs more RAM)"],
    index=0
)
llm_model_size = "gpt2-medium" if "gpt2-medium" in llm_choice else "distilgpt2"
caption_style = st.radio(
    "Caption style:",
    ["Classic", "Funny", "Poetic"],
    index=0,
    horizontal=True
)

upload = st.file_uploader("Upload an image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if upload:
    image = Image.open(upload).convert('RGB')
    st.image(image, caption="Your uploaded image", use_column_width=True)

    if st.button("Generate caption!"):
        with st.spinner("Analyzing and writing your creative caption..."):
            processor, blip_model = load_blip()
            tokenizer, llm_model = load_llm(llm_model_size)

            inputs = processor(images=image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            base_caption = processor.decode(out[0], skip_special_tokens=True)
            st.info(f"AI sees: _{base_caption}_")

            prompt_map = {
                "Classic": f"Write a catchy Instagram caption for this photo: {base_caption}",
                "Funny": f"Write a funny, witty Instagram caption for this photo: {base_caption}",
                "Poetic": f"Write a short, poetic caption for this image: {base_caption}",
            }
            prompt = prompt_map.get(caption_style, prompt_map["Classic"])

            new_caption = get_llm_caption(tokenizer, llm_model, prompt)

            st.success(f"**{caption_style} Caption:** {new_caption}")

            keywords_text = base_caption + " " + new_caption
            hashtags = extract_keywords_yake(keywords_text)
            style_hashtags = {
                "Classic": ["#PhotoOfTheDay", "#InstaGood"],
                "Funny": ["#LOL", "#GoodVibesOnly"],
                "Poetic": ["#ArtOfLife", "#Soulful"],
            }
            hashtags = list(dict.fromkeys(hashtags + style_hashtags[caption_style]))

            st.markdown("##### Hashtag Suggestions")
            st.code(" ".join(hashtags), language="")

            st.markdown("##### Bonus Caption Ideas")
            st.write(f"- {new_caption} {hashtags[0]}")
            st.write(f"- When words aren't enough... {hashtags[1]}")
            st.write(f"- Mood: {new_caption.lower()} {hashtags[-1]}")

st.caption("☁️ 100% local, all models run on your machine. Created for Outlier AI Playground.")

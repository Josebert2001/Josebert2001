import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import streamlit as st

@st.cache_resource
def load_model_and_tokenizer():
    try:
        model_name = "t5-large"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

class AIChatBot:
    def __init__(self):
        self.model_name = "t5-large"
        self.tokenizer, self.model = load_model_and_tokenizer()
        self.chat_history_ids = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model:
            self.model.to(self.device)

    def generate_response(self, user_input):
        try:
            if self.chat_history_ids is None:
                self.chat_history_ids = torch.empty(1, 0).to(self.device)

            input_ids = self.tokenizer.encode(
                user_input + self.tokenizer.eos_token, 
                return_tensors="pt"
            ).to(self.device)

            generation_params = {
                "max_length": 1000,
                "pad_token_id": self.tokenizer.eos_token_id,
                "no_repeat_ngram_size": 3,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                "decoder_start_token_id": self.tokenizer.pad_token_id
            }

            outputs = self.model.generate(
                input_ids,
                **generation_params
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return "Sorry, I encountered an issue. Please try again later."

def main():
    st.set_page_config(page_title="T5 Chat Bot", page_icon="🤖")
    st.title("T5 Chat Bot 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Limit chat history
    max_messages = 50
    if len(st.session_state.messages) > max_messages:
        st.session_state.messages = st.session_state.messages[-max_messages:]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        bot = AIChatBot()
        response = bot.generate_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    if st.session_state.messages:
        st.divider()
        feedback = st.radio("Rate the response:", ["Excellent", "Good", "Neutral", "Poor"])
        if feedback == "Poor":
            feedback_text = st.text_area("Additional feedback:")
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
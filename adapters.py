# adapters.py

class AWSAdapter:
    def convert_request(self, openai_messages):
        # Converts OpenAI [{"role": "user", "content": "..."}] 
        # to AWS Bedrock format (e.g., Llama 3 via Bedrock)
        prompt = ""
        for m in openai_messages:
            prompt += f"<|{m['role']}|>\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        
        return {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.5
        }

    def convert_response(self, bedrock_response):
        # Converts Bedrock JSON back to OpenAI JSON
        return {
            "id": "chatcmpl-aws-inf2",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": bedrock_response['generation']
                }
            }]
        }
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
import json
import time

class LocalQwenLLM:
    """Local Qwen LLM wrapper using Unsloth's 4-bit quantized model"""
    
    def __init__(self):
        print("🤖 Loading Local Qwen LLM (unsloth/Qwen3-14B-bnb-4bit)...")
        start_time = time.time()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-14B-bnb-4bit")
        
        # Load model with 4-bit quantization for RTX 4090 compatibility
        self.model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Qwen3-14B-bnb-4bit",
            torch_dtype="auto",
            device_map="auto"
        )
        
        load_time = time.time() - start_time
        print(f"✅ Local Qwen LLM loaded in {load_time:.2f} seconds")
    
    def complete(self, prompt: str) -> str:
        """Complete a prompt using the local Qwen model"""
        
        # Prepare the model input with chat template
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Enable thinking capability for better reasoning
        )
        
        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,  # Reasonable limit for entity extraction
                temperature=0.1,      # Low temperature for consistent output
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content if available
        try:
            # Find </think> token (151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        except ValueError:
            # No thinking content found
            thinking_content = ""
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        # Return the final content (after thinking)
        return content
    
    def extract_entities_and_relationships(self, document_text: str) -> Dict[str, Any]:
        """Extract entities and relationships using local Qwen model"""
        
        prompt = f"""
请从以下文档中提取实体和关系，以JSON格式返回：

文档内容：
{document_text}

请提取：
1. 实体（entities）：包括公司、角色、平台、车辆、流程、文档、地点、功能、要求、步骤等
2. 关系（relationships）：实体之间的关系，如REQUIRES、PROVIDES、BELONGS_TO、WORKS_FOR、HAS_STEPS、COMPETES_WITH、EARN、LOCATED_IN等

返回格式：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型", "properties": {{"key": "value"}}}}
    ],
    "relationships": [
        {{"source": "源实体", "target": "目标实体", "type": "关系类型", "properties": {{"key": "value"}}}}
    ]
}}

请确保返回的是有效的JSON格式，不要包含markdown代码块标记。
"""
        
        try:
            response = self.complete(prompt)
            
            # Clean response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]  # Remove ```json
            if response.startswith("```"):
                response = response[3:]   # Remove ```
            if response.endswith("```"):
                response = response[:-3]  # Remove ```
            
            response = response.strip()
            
            # Parse JSON response
            result = json.loads(response)
            return result
            
        except Exception as e:
            print(f"Error in local LLM graph generation: {e}")
            return {"entities": [], "relationships": []}

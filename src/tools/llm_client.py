from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")


class LLMClient:
    def __init__(self):
        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=API_KEY)
        self.default_model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

    def get_chat_model(
        self,
        prompt: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        guided_json: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Call NVIDIA NIM LLM with support for structured outputs via guided_json.
        
        Args:
            prompt: List of message dictionaries with 'role' and 'content'
            model: Model name (defaults to default_model)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 to 2.0)
            guided_json: JSON schema for structured output (enables guided_json)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dict with response data including 'content' field
        """
        model = model or self.default_model
        
        # Build the base request parameters
        request_params = {
            "model": model,
            "messages": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add extra_body parameters if guided_json is provided
        if guided_json is not None:
            request_params["extra_body"] = {"guided_json": guided_json}
        
        # Add any additional kwargs
        request_params.update(kwargs)
        
        # Make the API call
        response = self.client.chat.completions.create(**request_params)
        
        # Return response in a consistent format
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": response.usage,
            "raw_response": response
        }
    
    def get_default_model(self) -> str:
        """Get the default model name."""
        return self.default_model


if __name__ == "__main__":
    llm_client = LLMClient()
    
    # Example 1: Simple chat without structured output
    print("="*60)
    print("Example 1: Simple Chat")
    print("="*60)
    prompt = "Explain the steps to purchase a car insurance policy in India. Give the response in bullet points, step by step."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = llm_client.get_chat_model(messages)
    print("Response from LLM:")
    print(response["content"])
    print()
    
    # Example 2: With structured output (guided_json)
    print("="*60)
    print("Example 2: With Structured Output (guided_json)")
    print("="*60)
    from pydantic import BaseModel, Field
    import json
    
    class InsurancePlan(BaseModel):
        """Car insurance plan recommendation."""
        plan_name: str = Field(description="Name of the insurance plan")
        coverage_types: list[str] = Field(description="List of coverage types included")
        estimated_annual_premium: float = Field(description="Estimated annual premium in INR")
        deductible: float = Field(description="Deductible amount in INR")
        recommendation_reason: str = Field(description="Why this plan is recommended")
    
    schema = InsurancePlan.model_json_schema()
    
    messages = [
        {"role": "system", "content": "You are a car insurance advisor in India. Recommend suitable plans."},
        {"role": "user", "content": "I need a comprehensive car insurance plan for my new sedan in India."}
    ]
    
    response = llm_client.get_chat_model(
        messages,
        temperature=0.3,
        guided_json=schema
    )
    
    print("Structured Response:")
    response_data = json.loads(response["content"])
    plan = InsurancePlan(**response_data)
    print(f"  Plan: {plan.plan_name}")
    print(f"  Coverage: {', '.join(plan.coverage_types)}")
    print(f"  Premium: ₹{plan.estimated_annual_premium:.2f}")
    print(f"  Deductible: ₹{plan.deductible:.2f}")
    print(f"  Reason: {plan.recommendation_reason}")

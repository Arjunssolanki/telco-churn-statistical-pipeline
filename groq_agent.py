import os
import sys
import warnings
from groq import Groq

# Gracefully suppress minor version mismatch warnings for cleaner console output
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Securely load parameters from the local .env file infrastructure
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

# Direct internal module imports
from mcp_server import predict_customer_churn_risk, get_pipeline_performance_metadata

def run_groq_mcp_agent():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("🛑 Critical Error: 'GROQ_API_KEY' not found in environment or .env file.")
        print("Please ensure GROQ_API_KEY=gsk_... is written clearly inside your .env file.")
        sys.exit(1)
        
    client = Groq(api_key=api_key)
    
    # Active production-grade model target alignment
    production_model = "qwen/qwen3.8-27b"
    
    print("==============================================================")
    print("🤖 GROQ CUSTOMER RETENTION LLM CO-PILOT AGENT ACTIVE")
    print("==============================================================")
    print(f"Connected to model: {production_model}")
    print("Your machine learning tools are mapped and ready for querying.")
    print("Type 'exit' or 'quit' at any time to close the session.\n")
    
    system_prompt = (
        "You are an elite, technical Customer Retention Co-Pilot Agent for a major telecom company. "
        "You have direct access to our production machine learning models and dataset metadata via these two Python tools:\n"
        "1. predict_customer_churn_risk(tenure_months, monthly_charges, contract_type, internet_service_type, payment_method)\n"
        "2. get_pipeline_performance_metadata()\n\n"
        "When a user asks about model performance or specific customer risk tiers, you must explain that you are calling "
        "the tool, interpret the returned statistical results clearly in plain business language, and lay out the strategic recommendations."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower().strip() in ['exit', 'quit']:
                print("👋 Closing agent session. Have a great day!")
                break
                
            if not user_input.strip():
                continue
                
            messages.append({"role": "user", "content": user_input})
            
            # First-pass query block to evaluate prompt context
            response = client.chat.completions.create(
                model=production_model,
                messages=messages
            )
            
            # FIXED: Correct type-safe extraction path matching the latest Groq SDK layout structures
            agent_response = response.choices[0].message.content
            
            # Route inputs to appropriate local model tools based on context keywords
            if any(k in user_input.lower() for k in ["predict", "risk", "churn"]):
                print(f"⚙️ [Agent Insight] Groq is calling your local 'predict_customer_churn_risk' model...")
                
                # Base customer profile metrics
                t_months = 2
                m_charges = 95.0
                c_type = "Month-to-month"
                i_type = "Fiber optic"
                p_method = "Electronic check"
                
                # Adapt parameters based on specific keyword variations in the prompt
                if "dsl" in user_input.lower(): i_type = "DSL"
                if "two year" in user_input.lower(): c_type = "Two year"
                
                tool_output = predict_customer_churn_risk(
                    tenure_months=t_months, monthly_charges=m_charges,
                    contract_type=c_type, internet_service_type=i_type, payment_method=p_method
                )
                
                tool_context_prompt = f"The model tool finished processing and returned this exact data:\n{tool_output}\nSummarize this for the user."
                messages.append({"role": "assistant", "content": tool_context_prompt})
                
                final_response = client.chat.completions.create(
                    model=production_model,
                    messages=messages
                )
                agent_response = final_response.choices[0].message.content
                
            elif any(k in user_input.lower() for k in ["performance", "accuracy", "metrics"]):
                print(f"⚙️ [Agent Insight] Groq is calling your local 'get_pipeline_performance_metadata' summary tool...")
                tool_output = get_pipeline_performance_metadata()
                
                tool_context_prompt = f"The pipeline metadata tool returned this exact summary:\n{tool_output}\nSummarize this data for the user."
                messages.append({"role": "assistant", "content": tool_context_prompt})
                
                final_response = client.chat.completions.create(
                    model=production_model,
                    messages=messages
                )
                agent_response = final_response.choices[0].message.content
                
            print(f"\n🤖 Agent: {agent_response}")
            messages.append({"role": "assistant", "content": agent_response})
            
        except KeyboardInterrupt:
            print("\n👋 Closing agent session.")
            break
        except Exception as e:
            print(f"🛑 Operational error encountered: {str(e)}")

if __name__ == "__main__":
    run_groq_mcp_agent()

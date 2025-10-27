# Import required libraries
import os, json, re
import random
import asyncio
import pandas as pd, numpy as np
from tqdm import tqdm
from sharktank_utils import *
from openai import OpenAI
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

scenario_folder = './scenario_basic_deepseek_R1'
fact_path = 'facts_shark_tank_transcript_28_HummViewer.txt'
scenario_name = "_".join(fact_path.split('_')[1:])

# Read the facts
facts_store = load_facts()
# fact_dict = facts_store[fact_path]
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Define model array using Groq models
reference_models = [
    'llama-3.3-70b-versatile',           # High-performance, good for complex tasks
    'llama-3.1-8b-instant',              # Fast and reliable, good balance
    'qwen/qwen3-32b',          # Alternative high-performance model
    'qwen/qwen3-32b',              # Good for diverse tasks
    'openai/gpt-oss-120b'  # If available
]
aggregator_model = "llama-3.1-8b-instant"      # Fast aggregation
evaluator_model = "llama-3.3-70b-versatile"    # High-quality evaluation

# Set the system level prompt
NUM_AGENTS = 3
taskmaster_system_prompt = f"""
You are a sharktank pitch director. You will take control {NUM_AGENTS} other LLM agents like yourself. 
You will be given a fact_dict containing the key facts of the product you will be creating the pitch for.
Breakdown the task into 3 tasks and assign each portion to each LLM agent.
Output your answer in a messages list that can be parsed by python's `eval()`. For example: 
{{
    "agent1" : [{{"role":"system", "content":"<insert role>"}}, {{"role":"user","content":"<insert prompt>"}}],
    "agent2" : [{{"role":"system", "content":"<insert role>"}}, {{"role":"user","content":"<insert prompt>"}}],
    "agent3" : [{{"role":"system", "content":"<insert role>"}}, {{"role":"user","content":"<insert prompt>"}}],
}}
Create winning pitches! No hallucinations! No explanations of the pitch!
"""
aggregator_system_prompt  = """
You have been provided with a set of responses from various open-source models to the latest user query. 
Your task is to synthesize these responses into a single, high-quality response. Output a script for prospective entrepreneurs to use.
It is crucial to critically evaluate the information provided in these responses,
recognizing that some of it may be biased or incorrect. 
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. 
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models:"""

# Define the Groq client for use
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

def run_llm(model, system_prompt, user_prompt, temperature=0.7):
    """Run a single LLM call using Groq API."""
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error running LLM with model {model}: {e}")
        return f"Error: {e}"

def aggregate_results(results, original_prompt):
    """Aggregate multiple LLM responses into a single string."""
    aggregated = f"Original prompt: {original_prompt}\n\n"
    for i, result in enumerate(results, 1):
        aggregated += f"Response {i}:\n{result}\n\n"
    return aggregated

def generate_user_prompt(facts):
    return f"""
My name is Paul McCarthney
Here are the facts of my product: {facts}
"""


def aggregate_results(results, original_prompt):
    """Aggregate multiple LLM responses into a single string."""
    aggregated = f"Original prompt: {original_prompt}\n\n"
    for i, result in enumerate(results, 1):
        aggregated += f"Response {i}:\n{result}\n\n"
    return aggregated

def select_models(num_agents=NUM_AGENTS, reference_models=reference_models):
    return random.choices(reference_models, k=num_agents)

def generate_pitch(facts, num_agents=NUM_AGENTS, reference_models=reference_models):
    """Run the main loop of the MOA process."""
    prompt = generate_user_prompt(facts)
    
    # Get the LLM response
    llm_response = run_llm(
        model=aggregator_model,
        system_prompt=taskmaster_system_prompt,
        user_prompt=prompt
    )
    
    # Try to parse as JSON first, then fallback to eval if needed
    try:
        # Clean the response to extract JSON if it's wrapped in text
        import json
        import re
        
        # Look for JSON-like structure in the response
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            assignments = json.loads(json_str)
        else:
            # Fallback to eval for backward compatibility
            assignments = eval(llm_response)
            
    except (json.JSONDecodeError, SyntaxError) as e:
        print(f"Error parsing LLM response: {e}")
        print(f"LLM response was: {llm_response}")
        
        # Create a fallback assignment structure
        assignments = {
            "agent1": [{"role": "system", "content": "You are a pitch expert focusing on the problem and solution."}, 
                      {"role": "user", "content": f"Create a compelling pitch for: {prompt}"}],
            "agent2": [{"role": "system", "content": "You are a pitch expert focusing on market opportunity and business model."}, 
                      {"role": "user", "content": f"Create a compelling pitch for: {prompt}"}],
            "agent3": [{"role": "system", "content": "You are a pitch expert focusing on financial projections and ask."}, 
                      {"role": "user", "content": f"Create a compelling pitch for: {prompt}"}]
        }
    assignments_list = list(assignments.values())

    results = [run_llm(
        model=reference_models[i],
        system_prompt=assignments_list[i][0]['content'],
        user_prompt=assignments_list[i][1]['content']
    ) for i in range(num_agents)]

    final_pitch = run_llm(
        model=aggregator_model,
        system_prompt=aggregator_system_prompt,
        user_prompt=aggregate_results(results, prompt)
    )
    return final_pitch

# Create a fixed set of agents:
agents = select_models()

pitches = {}
for name, fact in facts_store.items():
    pitch = generate_pitch(
        fact, 
        num_agents=NUM_AGENTS, 
        reference_models=reference_models
    ).choices[0].message.content

    pitches[name] = pitch

# Output the dictionary as a JSON object
with open('basic_multiagent_pitches.json', 'w') as f:
    json.dump(pitches, f, indent=4)


editor_prompt = f"""
You are a pitch editor. You will be given a pitch. Evaluate its strength. 
Give constructive feedback on how to improve the pitch a for shark tank pitch.
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Here is the pitch:
"""

def generate_feedback_pitch(facts, num_agents=NUM_AGENTS, reference_models=reference_models, loops=3):
    """Run the main loop of the MOA process."""
    initial_pitch, feedback = None, None
    for l in tqdm(range(loops)):
        prompt = generate_user_prompt(facts)
        if initial_pitch:
            prompt += f"""
            Here was your previous attempt: {initial_pitch}. Improve upon it.
            """
        
        if feedback:
            prompt += f"""
            This is the feedback of your previous attempt: {feedback}
            """
        assignments = eval(run_llm(
            model=aggregator_model,
            system_prompt=taskmaster_system_prompt,
            user_prompt=prompt
        ))
        assignments_list = list(assignments.values())

        results = [
            run_llm(
                model=reference_models[i],
                system_prompt=assignments_list[i][0]['content'],
                user_prompt=assignments_list[i][1]['content']
            ) for i in range(num_agents)
        ]

        final_pitch = run_llm(
            model=aggregator_model,
            system_prompt=aggregator_system_prompt,
            user_prompt=aggregate_results(results, prompt)
        )

        if l < loops-1:
            evaluation = run_llm(
                model=evaluator_model,
                system_prompt=editor_prompt,
                user_prompt=final_pitch,
                temperature=0.7
            )

            initial_pitch = final_pitch
            feedback = evaluation
    return final_pitch


# Create a fixed set of agents:
agents = select_models()

pitches = {}
# for name, fact in facts_store:
sample = list(facts_store.keys())[0]
name, fact = sample, facts_store[sample]
pitch = generate_feedback_pitch(
    fact, 
    num_agents=NUM_AGENTS, 
    reference_models=reference_models
)
print(pitch)

pitches[name] = pitch
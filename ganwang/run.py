#!/usr/bin/env python3
"""
Simple runner script for the ganwang pitch generation scripts.
"""

import sys
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run pitch generation scripts')
    parser.add_argument('script', choices=['tavily', 'tavily-norag', 'ddg'], 
                       help='Which script to run')
    parser.add_argument('--product-key', type=str, 
                       default='facts_shark_tank_transcript_0_GarmaGuard.txt',
                       help='Product key to process')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='Model to use (for ddg script)')
    parser.add_argument('--specialist-model', type=str, default='gpt-4o',
                       help='Specialist model (for tavily scripts)')
    parser.add_argument('--pitch-model', type=str, default='gpt-4o',
                       help='Pitch model (for tavily scripts)')
    
    args, extra_args = parser.parse_known_args()
    
    script_map = {
        'tavily': 'async_specialist_team_single_rag_tool_tavily.py',
        'tavily-norag': 'async_specialist_team_single_rag_tool_tavily_norag.py',
        'ddg': 'async_specialist_team_single_rag_tool_ddg.py'
    }
    
    script_name = script_map[args.script]
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return 1
    
    # Build command
    cmd = [sys.executable, script_path, '--product-key', args.product_key]
    
    if args.script == 'ddg':
        cmd.extend(['--model', args.model])
    else:
        cmd.extend(['--specialist-model', args.specialist_model])
        cmd.extend(['--pitch-model', args.pitch_model])
    
    # Add any extra arguments
    cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run the script
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error running script: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())


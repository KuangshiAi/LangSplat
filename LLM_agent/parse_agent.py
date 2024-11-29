import openai
import subprocess
import shlex
import os

# Set your OpenAI API key
openai.api_key = 'Your_OpenAI_key'  # Make sure to set the OPENAI_API_KEY environment variable

def parse_user_input(user_input):
    """
    Parses the user input to extract positive queries, negative queries, and level indices.

    Returns:
        A dictionary with keys 'positive_queries', 'negative_queries', 'level_idx'.
    """
    # Define the function schema for OpenAI's function calling
    function = {
        "name": "parse_input",
        "description": ("Understand the user's natural language input to identify the items they want to include and exclude,"
                        " as well as any specific semantic levels they mention."
                        "positive_queries: A list of items the user wants to include."
                        "negative_queries: A list of items the user wants to exclude."
                        "level_idx: semantic levels ('1': Subpart, '2': Part, '3': Whole segmentation)."
                        "Look for phrases like 'show me', 'visualize' for inclusion; 'remove', 'hide' for exclusion; and 'subpart', 'part', 'whole', or level numbers for level_idx."
                        " If levels aren't specified, default to ['1', '2', '3']."
                        "Input: 'Show me the heart and lungs, but exclude the liver. I want to see the subpart and whole segmentation.'"
                        "Output: {'positive_queries': ['heart', 'lungs'], 'negative_queries': ['liver'], 'level_idx': ['1', '3']}"
                        "Input: Show me the toothpaste and remove the water bottle"
                        "Output: {'positive_queries': ['toothpaste'], 'negative_queries': ['water bottle'], 'level_idx': ['1', '2', '3']}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "positive_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of items the user wants to include."
                },
                "negative_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of items the user wants to exclude."
                },
                "level_idx": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of level indices for visualization."
                }
            },
            "required": ["positive_queries", "negative_queries", "level_idx"]
        }
    }

    messages = [
        {"role": "system", "content": "You are an AI assistant that helps to parse user input for visualization."},
        {"role": "user", "content": user_input}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        functions=[function],
        function_call={"name": "parse_input"},
        temperature=0,
    )

    # Extract the arguments from the function call
    function_call = response['choices'][0]['message']['function_call']
    arguments = function_call['arguments']

    import json

    try:
        parsed_arguments = json.loads(arguments)
        # Set default level indices if not specified
        if not parsed_arguments.get('level_idx'):
            parsed_arguments['level_idx'] = ["1", "2", "3"]
        return parsed_arguments
    except json.JSONDecodeError:
        print("Error: Could not parse assistant's response as JSON.")
        return None

def construct_command(args):
    # Construct the command to run the evaluate_scivis.py script
    python_command = [
        "python", "evaluate_scivis.py",
        "--dataset_name", "flare_label",
        "--feat_dir", "/home/kuangshiai/Documents/LangSplat-results/output",
        "--ae_ckpt_dir", "/home/kuangshiai/Documents/LangSplat-results/autoencoder_ckpt",
        "--output_dir", "/home/kuangshiai/Documents/LangSplat-results/eval_result",
        "--mask_thresh", "0.4",
        "--encoder_dims", "256", "128", "64", "32", "3",
        "--decoder_dims", "16", "32", "64", "128", "256", "256", "512",
        "--json_folder", "\"\"",
        "--test_idx", "00001", "00004", "00005", "00006", "00008", "00009", "00014",
        "--positive_queries"
    ]
    # Add "backpack" as default negative query
    # args['negative_queries'].append('"backpack"')
    args['positive_queries'] = ['"' + query + '"' for query in args['positive_queries']]
    args['negative_queries'] = ['"' + query + '"' for query in args['negative_queries']]

    python_command.extend(args['positive_queries'])

    # Add negative queries
    python_command.append("--negative_queries")
    python_command.extend(args['negative_queries'] if args['negative_queries'] else [""])

    # Add level indices
    python_command.append("--level_idx")
    python_command.extend(args['level_idx'])

    # Join everything for the final command
    full_command = f"cd .. && cd eval && {' '.join(python_command)}"
    return full_command

def main():
    user_input = input("Please enter your request: ")
    args = parse_user_input(user_input)
    if args is None:
        print("Could not parse user input.")
        return

    command = construct_command(args)
    print("Running command:", command)
    print("\nProcessing...\n")

    try:
        # Execute the command in a shell
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return

    # Provide a summary to the user
    print("\nSummary:")
    print(f"Positive queries: {args['positive_queries']}")
    print(f"Negative queries: {args['negative_queries']}")
    print(f"Semantic levels used: {args['level_idx']}")

if __name__ == "__main__":
    main()

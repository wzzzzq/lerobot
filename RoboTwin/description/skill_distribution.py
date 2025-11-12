import json
import os
# import re
# import nltk
# from collections import Counter
from typing import Dict, List, Set, Tuple, Optional
import matplotlib.pyplot as plt
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
import yaml
import pdb

curent_path = os.path.dirname(os.path.abspath(__file__))
# Download necessary NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('taggers/averaged_perceptron_tagger')
# except LookupError:
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

# # Define a set of common verbs that are not robotic skills
# NON_SKILL_VERBS = {
#     'do', 'does', 'did', 'done', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
#     'have', 'has', 'had', 'having', 'can', 'could', 'will', 'would', 'shall', 'should',
#     'may', 'might', 'must', 'ought', 'need', 'dare', 'used', 'get', 'gets', 'got', 'gotten',
#     'getting', 'go', 'goes', 'went', 'gone', 'going', 'make', 'makes', 'made', 'making',
#     'let', 'lets', 'letting', "use", "uses", "used", "using", "take", "takes", "took", "taken", "taking",
#     "origin", "return", "align", "generate", "generated", "generating", "versa", "left", "right", "simulate", "complete.do", "wait",
#     "object.place", "origin.note", "selected", "fp", "set", "away.grasp", "stand.grasp", "position.do", "block.grsp", "put", "first", "pass",
#     "color", "colored", "'t", "beside", "upward", "downward", "forward", "backward", "leftward", "rightward", "upwards", "downwards", "specified",
#     "decrease", "increase", "object", "pose", "front", "call", "lie", "arrange", "smallest", "lying"
# }


# SKILL_CHANGE = {
#     "placed": "place",
#     "placing": "place",
#     "beating": "beat",
#     "lying": "lie",
#     "grabbing": "grasp",
#     "placing": "place",
#     "needs": "need",
#     "closed": "close",
#     "clicking": "click",
#     "moving": "move",
#     "specified": "specify",
#     "hanging": "hang",
#     "pressing": "press",
#     "grasping": "grasp",
#     "pouring": "pour",
#     "pressed": "press",
#     "lifting": "lift",
#     "grab": "grasp",
#     "catch": "grasp"
# }

def loadd_yml_data(file_path: str) -> Dict:
    """
    Load YAML data from a file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Parsed YAML data as a dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_json_data(file_path: str) -> Dict:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data as a dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# def extract_verbs_from_instruction(instruction: str) -> List[str]:
#     """
#     Extract unique verbs from a language instruction.
    
#     Args:
#         instruction: A string containing the language instruction
        
#     Returns:
#         A list of unique verbs found in the instruction
#     """
#     # Tokenize the text
#     tokens = word_tokenize(instruction.lower())
    
#     # Perform part-of-speech tagging
#     tagged_tokens = pos_tag(tokens)
    
#     # Extract words tagged as verbs (VB, VBD, VBG, VBN, VBP, VBZ)
#     # Filter out common non-skill verbs
#     verbs = [word for word, tag in tagged_tokens if tag.startswith('VB') and len(word) > 1 and word not in NON_SKILL_VERBS]
#     print(instruction, verbs)
#     # Return unique verbs
#     return list(set(verbs))

# def analyze_skill_distribution(task_description, output_dir: Optional[str] = None) -> Dict:
#     """
#     Analyze the skill distribution from language instruction files.
    
#     Args:
#         data_dir: Directory containing language instruction JSON files
#         output_dir: Optional directory to save visualization results
        
#     Returns:
#         A dictionary containing skill distribution statistics
#     """
#     all_verbs = []
#     file_count = 0
    
#     # pdb.set_trace()
#     # Process all JSON files in the directory
#     # for filename in os.listdir(data_dir):
#     task_verbs = {}
#     all_skill_ave_success_rate = {}
#     for task in task_description.keys():
#         description = task_description[task]['task_description']
#         description = description[:description.find('Note:')]
#         # delete all '+', '=', '-' characters
#         description = re.sub(r'[+=-]', '', description)
#         # 删除被空格隔开的部分中，. 前后都是字母的内容
#         description = re.sub(r'\b\w+\.\w+\b', '', description)
#         # delete all open gripper and close grippe
#         description = re.sub(r'\b(open|close) gripper\b', '', description)
#         description = re.sub(r'\b(open|close) the gripper\b', '', description)
#         verbs = extract_verbs_from_instruction(description)
#         res_verbs = []
#         for verb in verbs:
#             if verb in SKILL_CHANGE:
#                 new_verb = SKILL_CHANGE[verb]
#                 if new_verb not in res_verbs:
#                     res_verbs.append(new_verb)
#             else:
#                 if verb not in res_verbs:
#                     res_verbs.append(verb)
#         verbs = res_verbs
#         all_verbs.extend(verbs)
#         file_count += 1
#         task_verbs[task] = verbs
#         for verb in verbs:
#             if verb not in all_skill_ave_success_rate:
#                 all_skill_ave_success_rate[verb] = []
#             if 'success_rate' in task_description[task]:
#                 all_skill_ave_success_rate[verb].append(task_description[task]['success_rate'])
#     all_skill_ave_success_rate = {k: sum(v) / len(v) if v else 0 for k, v in all_skill_ave_success_rate.items()}
#     # sort the all_skill_ave_success_rate by value
#     all_skill_ave_success_rate = dict(sorted(all_skill_ave_success_rate.items(), key=lambda item: item[1], reverse=True))
#         # if filename.endswith('.json'):
#         #     file_path = os.path.join(data_dir, filename)
#         #     try:
#         #         with open(file_path, 'r', encoding='utf-8') as f:
#         #             data = json.load(f)
                    
#         #             # Extract full description from the JSON
#         #             if 'full_description' in data:
#         #                 description = data['full_description']
                        
#         #                 # Extract verbs from the description
#         #                 verbs = extract_verbs_from_instruction(description)
#         #                 all_verbs.extend(verbs)
#         #                 file_count += 1
#         #     except Exception as e:
#         #         print(f"Error processing {filename}: {e}")
    
#     # Count verb frequencies
#     verb_counter = Counter(all_verbs)
#     print(task_verbs)
    
#     # Calculate statistics
#     total_verbs = len(all_verbs)
#     unique_verbs = len(verb_counter)
    
#     # Prepare results
#     results = {
#         "total_files": file_count,
#         "total_verbs": total_verbs,
#         "unique_verbs": unique_verbs,
#         # "verb_distribution": dict(verb_counter.most_common()),
#         "verb_distribution": {k: v for k, v in all_skill_ave_success_rate.items()},
#         "average_verbs_per_instruction": total_verbs / file_count if file_count > 0 else 0,
#         "task_verbs": task_verbs
#     }
#     with open(os.path.join(output_dir, 'task_verbs.json'), 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4)
    
#     # Generate visualization if output directory is provided
#     if output_dir and os.path.exists(output_dir):
#         visualize_skill_distribution(results, output_dir)
    
#     return results

def visualize_skill_distribution(results: Dict, output_dir: str) -> None:
    """
    Visualize the skill distribution and save the plots.
    
    Args:
        results: Dictionary containing skill distribution statistics
        output_dir: Directory to save visualization results
    """
    # Plot verb frequency distribution
    plt.figure(figsize=(12, 8))
    
    # Use top 50 verbs for visualization
    all_verbs = dict(list(results["verb_distribution"].items())[:50])
    
    plt.bar(all_verbs.keys(), all_verbs.values())
    plt.xlabel('Verbs')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.title('Verbs by Success Rate')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'verb_distribution.png'))
    plt.close()
    
    # Save statistics as JSON
    with open(os.path.join(output_dir, 'skill_distribution_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

# def get_skill_complexity(instruction_file: str) -> int:
#     """
#     Get the skill complexity of a single instruction file by counting unique verbs.
    
#     Args:
#         instruction_file: Path to the instruction JSON file
        
#     Returns:
#         Number of unique verbs (skill complexity)
#     """
#     try:
#         with open(instruction_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
            
#             if 'full_description' in data:
#                 description = data['full_description']
#                 verbs = extract_verbs_from_instruction(description)
#                 return len(verbs)
#     except Exception as e:
#         print(f"Error processing {instruction_file}: {e}")
    
#     return 0

# def analyze_seen_unseen_distribution(data_dir: str, output_dir: Optional[str] = None) -> Dict:
#     """
#     Analyze the skill distribution from seen and unseen instructions.
    
#     Args:
#         data_dir: Directory containing language instruction JSON files
#         output_dir: Optional directory to save visualization results
        
#     Returns:
#         A dictionary containing skill distribution statistics for seen and unseen instructions
#     """
#     seen_verbs = []
#     unseen_verbs = []
#     file_count = 0
    
#     # Process all JSON files in the directory
#     for filename in os.listdir(data_dir):
#         if filename.endswith('.json'):
#             file_path = os.path.join(data_dir, filename)
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
                    
#                     # Process seen instructions
#                     if 'seen' in data and isinstance(data['seen'], list):
#                         for instruction in data['seen']:
#                             verbs = extract_verbs_from_instruction(instruction)
#                             seen_verbs.extend(verbs)
                    
#                     # Process unseen instructions
#                     if 'unseen' in data and isinstance(data['unseen'], list):
#                         for instruction in data['unseen']:
#                             verbs = extract_verbs_from_instruction(instruction)
#                             unseen_verbs.extend(verbs)
                    
#                     file_count += 1
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")
    
#     # Count verb frequencies
#     seen_verb_counter = Counter(seen_verbs)
#     unseen_verb_counter = Counter(unseen_verbs)
    
#     # Calculate statistics
#     seen_total_verbs = len(seen_verbs)
#     seen_unique_verbs = len(seen_verb_counter)
    
#     unseen_total_verbs = len(unseen_verbs)
#     unseen_unique_verbs = len(unseen_verb_counter)
    
#     # Prepare results
#     results = {
#         "total_files": file_count,
#         "seen": {
#             "total_verbs": seen_total_verbs,
#             "unique_verbs": seen_unique_verbs,
#             "verb_distribution": dict(seen_verb_counter.most_common()),
#             "average_verbs_per_instruction": seen_total_verbs / len(seen_verbs) if seen_verbs else 0
#         },
#         "unseen": {
#             "total_verbs": unseen_total_verbs,
#             "unique_verbs": unseen_unique_verbs,
#             "verb_distribution": dict(unseen_verb_counter.most_common()),
#             "average_verbs_per_instruction": unseen_total_verbs / len(unseen_verbs) if unseen_verbs else 0
#         }
#     }
    
#     # Generate visualization if output directory is provided
#     if output_dir and os.path.exists(output_dir):
#         visualize_seen_unseen_distribution(results, output_dir)
    
#     return results

# def visualize_seen_unseen_distribution(results: Dict, output_dir: str) -> None:
#     """
#     Visualize the skill distribution for seen and unseen instructions and save the plots.
    
#     Args:
#         results: Dictionary containing skill distribution statistics
#         output_dir: Directory to save visualization results
#     """
#     # Plot seen verb frequency distribution
#     plt.figure(figsize=(12, 8))
#     seen_verbs = dict(list(results["seen"]["verb_distribution"].items())[:50])
#     plt.bar(seen_verbs.keys(), seen_verbs.values(), color='blue')
#     plt.xlabel('Verbs')
#     plt.ylabel('Frequency')
#     # plt.title('Top 50 Verbs in Seen Instructions')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'seen_verb_distribution.png'))
#     plt.close()
    
#     # Plot unseen verb frequency distribution
#     plt.figure(figsize=(12, 8))
#     unseen_verbs = dict(list(results["unseen"]["verb_distribution"].items())[:50])
#     plt.bar(unseen_verbs.keys(), unseen_verbs.values(), color='red')
#     plt.xlabel('Verbs')
#     plt.ylabel('Frequency')
#     # plt.title('Top 50 Verbs in Unseen Instructions')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'unseen_verb_distribution.png'))
#     plt.close()
    
    # Save statistics as JSON
    with open(os.path.join(output_dir, 'seen_unseen_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Define paths
    # data_dir = f"{curent_path}/./tasks_description.yml"
    result_dir = f"{curent_path}/results.json"
    output_dir = f"{curent_path}/output/"
    results = load_json_data(result_dir)
    if output_dir and os.path.exists(output_dir):
        visualize_skill_distribution(results, output_dir)
    # task_description = loadd_yml_data(data_dir)
    # # Create output directory if it doesn't exist
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    # # Analyze overall skill distribution
    # print("Analyzing overall skill distribution...")
    # results = analyze_skill_distribution(task_description, output_dir)
    # print(f"Found {results['unique_verbs']} unique verbs across {results['total_files']} files")
    # print(f"Average verbs per instruction: {results['average_verbs_per_instruction']:.2f}")
    
    # Analyze seen vs unseen distribution
    # print("\nAnalyzing seen vs unseen instruction distribution...")
    # seen_unseen_results = analyze_seen_unseen_distribution(data_dir, output_dir)
    # print(f"Seen instructions: {seen_unseen_results['seen']['unique_verbs']} unique verbs")
    # print(f"Unseen instructions: {seen_unseen_results['unseen']['unique_verbs']} unique verbs")
    
    # print(f"\nResults saved to {output_dir}")

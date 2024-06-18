import argparse, os, ast
import ast2vec
import python_ast_utils
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Script for running local LLM with few-shot learning for discovering the parameters for speedupy")
    parser.add_argument("-m", "--model", type=str, default="gpt2", help="LLM name")
    parser.add_argument("-p", "--prompt", type=str, default="prompt.txt", help="prompt template file")
    parser.add_argument("-d", "--dir", type=str, help="directory containing the scripts for the few-shot learning. The scripts inputs should be contained in this directory (inputs.txt), as well as the values for the parameters (outputs.txt)", required=True)
    parser.add_argument("args", type=str, help="directory containing the script and input for the script. Should contain the script and inputs.txt with the inputs of the script", nargs=1)
    return parser.parse_args()    

def get_model(args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )

def read_file(file):
    with open(file, "r") as f:
        return f.read()

def read_dir(dir):
    scripts = []
    inputs = None
    outputs = None
    for file in os.listdir(dir):
        if file == "inputs.txt":
            with open(os.path.join(dir, file), "r") as f:
                inputs = f.readlines()
            continue
        if file == "outputs.txt":
            with open(os.path.join(dir, file), "r") as f:
                outputs = f.readlines()
            continue
        with open(os.path.join(dir, file), "r") as f:
            scripts.append(f.read())
    
    return scripts, inputs, outputs

def get_ast(script):
    return python_ast_utils.ast_to_tree(ast.parse(script))

def get_code2vec(script):
    tree = python_ast_utils.ast_to_tree(ast.parse(script))
    model = ast2vec.load_model()
    return model.encode(tree)

def get_examples(scripts, inputs, outputs=None):
    if outputs is None:
        outputs = [""]*len(scripts)
    return ('\n\n' + '-'*50 + '\n\n').join([f"input: {input}\n\nscript: {get_code2vec(script)} \n\noutput: {output}" for script, input, output in zip(scripts, inputs, outputs)])
    # return ('\n\n' + '-'*50 + '\n\n').join([f"input: {input}\n\nscript: {script} \n\noutput: {output}" for script, input, output in zip(scripts, inputs, outputs)])
    # return ('\n\n' + '-'*50 + '\n\n').join([f"input: {input}\n\nscript: {get_ast(script)} \n\noutput: {output}" for script, input, output in zip(scripts, inputs, outputs)])


def main():
    print(torch.cuda.is_available())
    login()
    args = parse_args()
    generator = get_model(args)
    prompt = read_file(args.prompt)
    scripts, inputs, outputs = read_dir(args.dir)
    script_final, input_final, _ = read_dir(args.args[0])
    prompt = prompt.format(examples=get_examples(scripts, inputs), question=get_examples(script_final, input_final))
    result = generator(
                    prompt, 
                    do_sample=True,
                    max_new_tokens=6000, 
                    temperature=0.7, 
                    top_k=50, 
                    top_p=0.95,
                )
    print(result) 
    # print(prompt)

if __name__ == "__main__":
    main()
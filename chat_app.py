import torch
import torch.nn.functional as F
from tokenizer import Tokenizer, ChatFormat
from llama import Llama, device
torch.set_default_dtype(torch.bfloat16)
path = '/home/jaimeet/.llama/checkpoints/Llama3.2-1B-Instruct'

tok = Tokenizer(path + '/tokenizer.model')
chat = ChatFormat(tok)

print('Hi, Welcome to Beautiful LLama!')
system_prompt = input('Please provide a system instruction to beautiful llama\nSystem: ')
dialog = [
    {
        "role": "system",
        "content": system_prompt,
    },
]
input_tok = torch.tensor([chat.encode_dialog_prompt_without_assistant(dialog)], device=device)

while True:
    user_prompt = input('User: ')
    user_message = {"role": "user", "content": user_prompt}
    user_tok_list = chat.encode_message(user_message) # Encode the user message with <|eot_id|>
    user_tok = torch.tensor([user_tok_list], device=device)
    input_tok = torch.cat((input_tok, user_tok), dim=1)

    # Add the assistant start header
    assistant_header = chat.encode_header({"role": "assistant", "content": ""})
    assistant_header_tok = torch.tensor([assistant_header], device=device)
    input_tok = torch.cat((input_tok, assistant_header_tok), dim=1)
    print("Assistant: ", end="")
    with torch.no_grad():
        while True:
            output = Llama(input_tok)
            logits = output[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            input_tok = torch.cat((input_tok, xcol), dim=-1)
            if xcol.item() in tok.stop_tokens:
                print("\n")
                break
            print(tok.decode([xcol.item()]), end="")
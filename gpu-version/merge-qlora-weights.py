from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import argparse

def main(peft_model_path: str, output_path: str):
    print(f"🔄 Loading PEFT model from: {peft_model_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder="./offload"
    )

    print("🧬 Merging adapter weights into the base model...")
    model = model.merge_and_unload()

    print("💾 Saving merged model to:", output_path)
    model.save_pretrained(output_path, safe_serialization=True)

    print("🔎 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    tokenizer.save_pretrained(output_path)

    print("✅ Merge complete. You can now push the model to Hugging Face!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("peft_model_path", help="Path to the PEFT fine-tuned model directory")
    parser.add_argument("output_path", help="Path to save the merged model")
    args = parser.parse_args()
    main(args.peft_model_path, args.output_path)

"""
Fine-tuning script for kri kri
"""
import os
from dataclasses import (
    dataclass,
    field
)
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from peft import (
        AutoPeftModelForCausalLM,
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from datasets import load_from_disk
import torch
import torch.distributed as dist
import bitsandbytes as bnb
import deepspeed
from huggingface_hub import login


# upgrade flash attention here
try:
    os.system("pip install flash-attn --no-build-isolation --upgrade")
except:
    print("flash-attn failed to install")



def find_all_linear_names(model):
    """
    Find modules to apply lora (standard qlora implementation)
    """
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)



def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    """
    create quantized model and lora adapters
    """
    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # pre-process the model by upcasting the layer norms in float 32 for
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model


def training_function(script_args, training_args):
    """
    declare training function
    """

    # Environment variables set by torch.distributed.launch
    ## local_rank = int(os.environ['LOCAL_RANK'])
    ## model_name_or_path = "/tmp/llama_source/"
    ## Download source model from S3 using s5cmd for local rank 0
    ## if LOCAL_RANK == 0:
    ##     print("-----------local rank 0 downloading model from s3----")
    ##     os.system("./s5cmd sync {0} {1}".format(
    ## os.environ['SOURCE_MODEL_BEFORE_TRAINING_S3_PATH'], model_name_or_path))
    ## Note: the barrier is used to ensure just only local rank 0 to download model assets from s3.
    ## torch.distributed.barrier()

    world_size = int(os.environ['WORLD_SIZE'])
    world_rank = int(os.environ['RANK'])

    # initialize process group
    dist.init_process_group(
        backend=script_args.distributed_backend,
        rank=world_rank,
        world_size=world_size
    )

    # load dataset
    dataset = load_from_disk(script_args.dataset_path)

    # load model from the hub with a bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load pretrained model in quantization state + flash attention 2 if chosen
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map="auto",
        use_flash_attention_2=script_args.use_flash_attn,
        quantization_config=bnb_config
    )

    # create peft config
    model = create_peft_model(
        model,
        gradient_checkpointing=training_args.gradient_checkpointing,
        bf16=training_args.bf16
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator
    )

    # Start training
    trainer.train()

    sagemaker_save_dir = "/opt/ml/model/"
    if script_args.merge_adapters:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            training_args.output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # Merge LoRA and base model and save
        model = model.merge_and_unload()
        model.save_pretrained(sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB")
    else:
        trainer.model.save_pretrained(sagemaker_save_dir, safe_serialization=True)

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, padding_side="left")
    tokenizer.save_pretrained(sagemaker_save_dir)

    ## We now save the model assets to an intermediate path.
    ## Note: plesae do not save the model into /opt/ml/model
    ## (because Sagemaker will tar and compress
    ## all of files under /opt/ml/model,
    ## and it will consume much time for LLM.)
    # print("------saving model!-----")
    # save_model_dir = '/tmp/output/asset/'
    # tokenizer.save_pretrained(save_model_dir)
    # trainer.save_model(save_model_dir)
    ## print("-----model is saved!----")
    ## Note: we just use the rank 0 process to upload the
    ## trained model assets to S3 by s5cmd command.
    # if WORLD_RANK == 0:
    #     os.system("./s5cmd sync {0} {1}".format(
    # save_model_dir, 
    # os.environ['TARGET_MODEL_AFTER_TRAINING_S3_PATH']
    #   )
    # )
    ## Note: we should sync with every ranker and ensure only global rank 0
    ## uploading the model assets successfully.
    # torch.distributed.barrier()


@dataclass
class ScriptArguments:
    """
    set script arguments
    """
    model_id: str = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub."
        },
    )
    dataset_path: str = field(
        metadata={
            "help": "Path to the preprocessed and tokenized dataset."
        },
        default=None,
    )
    hf_token: Optional[str] = field(
        metadata={
            "help": "Hugging Face token for authentication"
        },
        default=None
    )
    trust_remote_code: bool = field(
        metadata={
            "help": "Whether to trust remote code."
        },
        default=False,
    )
    distributed_backend: Optional[str] = field(
        metadata={
            "help": "Distributed backed to use"
        },
        default=None
    )
    use_flash_attn: bool = field(
        metadata={
            "help": "Whether to use Flash Attention."
        },
        default=False,
    )
    merge_adapters: bool = field(
        metadata={
            "help": "Whether to merge weights for LoRA."
        },
        default=False,
    )


def main():
    """
    run training script
    """

    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    # set seed
    set_seed(training_args.seed)

    # login to hub
    token = script_args.hf_token if script_args.hf_token else os.getenv("HF_TOKEN", None)
    if token:
        print(f"Logging into the Hugging Face Hub with token {token[:10]}...")
        login(token=token)

    # run training function
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()

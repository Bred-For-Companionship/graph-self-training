from loader_utils.prompt_jinja_template import llava13b_template

class LLavaDataCollatorCompletionsOnly:
    def __init__(self, processor):
        self.processor = processor

        if '7b' in processor.tokenizer.name_or_path:
            self.version = '7b'  ##different versions use different input format; 34b not supported yet
        else:
            self.version = '13b'
            self.processor.tokenizer.chat_template = llava13b_template
        self.MAX_LENGTH = 2048

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            if len(example["images"]) > 1:
                raise ValueError("This collator only supports one image per example")
            text = example["conversations"]
            text = self.processor.tokenizer.apply_chat_template(
                text, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["image"])

        batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=self.MAX_LENGTH, return_tensors="pt")

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
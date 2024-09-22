from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load a pre-trained model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Function to fine-tune T5 model with new feedback data
def fine_tune_model(feedback_texts, summaries):
    inputs = tokenizer(feedback_texts, return_tensors='pt', padding=True, truncation=True)
    labels = tokenizer(summaries, return_tensors='pt', padding=True, truncation=True)
    
    model.train()
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Fine-tuning loop (simplified)
    for epoch in range(3):  # Adjust epoch as needed
        optimizer.zero_grad()
        outputs = model(input_ids=inputs.input_ids, labels=labels.input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Save the fine-tuned model
    model.save_pretrained('./fine-tuned-model')
    tokenizer.save_pretrained('./fine-tuned-model')

# Example call to fine-tune the model
feedback_data = ["Employee is dissatisfied with project deadlines.", "Manager is doing great work!"]
summaries = ["Dissatisfaction with deadlines", "Positive feedback for manager"]

fine_tune_model(feedback_data, summaries)

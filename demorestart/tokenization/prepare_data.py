from huggingface_hub import hf_hub_download

train_file = hf_hub_download(
    repo_id="roneneldan/TinyStories",
    filename="TinyStories-train.txt",
    repo_type="dataset",
    local_dir="data"
)

val_file = hf_hub_download(
    repo_id="roneneldan/TinyStories",
    filename="TinyStories-valid.txt",
    repo_type="dataset",
    local_dir="data"
)

print("Train:", train_file)
print("Validation:", val_file)
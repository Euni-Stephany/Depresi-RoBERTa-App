from huggingface_hub import create_repo, upload_folder

username = "Eunii" 
repo_name = "Depresi-RoBERTa-App" 

upload_folder(
    folder_path="content/model_mental_roberta", 
    path_in_repo=".",  # Upload semua ke root
    repo_id=f"{username}/{repo_name}",
    repo_type="model",
    commit_message="first commit"
)
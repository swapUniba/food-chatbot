# food-chatbot
A digital assistant about food that also serves as a food recommender chatbot and is especially concerned with sustainability and healthiness. It comes in two versions: one English, based on LLaMA, and one Italian, based on LLaMAntino.

## How to setup the project for running the code on a Windows machine (if you're already using a machine with Ubuntu-20.04 skip the following steps)
- Make sure the NVIDIA and CUDA drivers are installed on the machine (if not, refer to https://developer.nvidia.com/cuda-12-3-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local and https://www.nvidia.com/download/index.aspx)
- Open Windows Features
- Enable the option "windows subsystem for Linux"
- Open the cmd as administrator
- Run the following command "wsl --install -d Ubuntu-20.04"
- Open the Ubuntu-20.04
- Set up a username and a password

## How to set up Ubuntu-20.04 for NVIDIA 
- Type "sudo apt-key del 7fa2af80" to remove the old GPG key
- Type "wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin"
- Type "sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600"
- Type "wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb"
- Type "sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.2-1_amd64.deb"
- Type "sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/"
- Type "sudo apt-get update"
- Type "sudo apt-get -y install cuda-toolkit-12-3"

## How to create a new session with a python 3.8 environment 
- Type "tmux new-session -t llamantino" 
- Type "sudo apt install python3.8-venv"
- Type "python3 -m venv llamantinoENV" (this command will likely install the environment inside the path "\home\your_linux_username\")
- Type "source llamantinoENV/bin/activate"

## How to install packages
- Type "pip install transformers"
- Type "pip install accelerate"
- Type "pip install streamlit"
- Type "pip install gradio"
- Type "pip install emoji"
- Type "pip install bitsandbytes"
- Type "pip install huggingface-hub"
- Type "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
- Due to a problem with the current version of the guidance library, as a workaround, install the older version by copying the "guidance" and "guidance-0.1.10.dist-info" directories inside the \home\your_linux_username\llamantinoENV\lib\python3.8\site-packages directory manually

## How to setup the code
- Copy the "food_chatbot", "Italian_chatbot_log" and "English_chatbot_log" directories inside the directory "\home\your_linux_username\"

## How to execute the Italian chatbot
- Type "tmux attach-session -t llamantino" if you're not already inside the session
- Type the command python "/home/your_linux_username/food-chatbot/my_gradio_app_ita.py"

## How to execute the English chatbot
- Open the code in any IDE and substitute the access_token_read value with your huggingface access token to login into Huggingface (the account must have access to the family of models LLaMA2) and save. If you don't have it, you can create a token on Huggingface (THIS IS TO DO ONLY THE FIRST TIME)
- Type "tmux attach-session -t llamantino" if you're not already inside the session
- Type the command python "/home/your_linux_username/food-chatbot/my_gradio_app.py"
